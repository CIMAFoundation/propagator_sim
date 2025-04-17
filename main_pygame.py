#!/usr/bin/env python3
import sys
import logging
import numpy as np
try:
    import pygame
except ImportError:
    print("pygame is required to run this simulation. Please install pygame: pip install pygame")
    sys.exit(1)

from propagator.functions import moist_proba_correction_1, p_time_wang
from propagator.loader.geotiff import PropagatorDataFromGeotiffs
from propagator.propagator import Propagator, PropagatorBoundaryConditions
from propagator.logging_config import configure_logger

def main():
    configure_logger()
    v0 = np.loadtxt("v0_table.txt")
    prob_table = np.loadtxt("prob_table.txt")
    p_veg = np.loadtxt("p_vegetation.txt")

    # Load input data
    loader = PropagatorDataFromGeotiffs(
        dem_file="example/dem_clip.tif",
        veg_file="example/veg_clip.tif",
    )
    dem = loader.get_dem()
    veg = loader.get_veg()

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=100,
        ros_0=v0,
        probability_table=prob_table,
        veg_parameters=p_veg,
        do_spotting=False,
        p_time_fn=p_time_wang,
        p_moist_fn=moist_proba_correction_1,
    )

    # Default environmental conditions
    wind_speed_scalar = 10.0
    wind_dir_scalar = 180.0
    wind_speed = np.ones(dem.shape) * wind_speed_scalar
    wind_dir = np.ones(dem.shape) * wind_dir_scalar
    moisture = np.ones(dem.shape) * 0.05

    # Prepare visualization
    H, W = veg.shape
    # Color map for vegetation types
    veg_colors = {
        0: (0, 0, 0),
        1: (34, 139, 34),
        2: (50, 205, 50),
        3: (222, 184, 135),
        4: (139, 69, 19),
        5: (34, 100, 34),
        6: (107, 142, 35),
    }
    veg_color_map = np.zeros((H, W, 3), dtype=np.uint8)
    for code, color in veg_colors.items():
        veg_color_map[veg == code] = color

    # Pygame setup
    pygame.init()
    cell_size = 4
    # Menu panel width
    menu_width = 200
    screen_width = W * cell_size + menu_width
    screen_height = H * cell_size
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    pygame.display.set_caption("Wildfire Simulator")
    clock = pygame.time.Clock()
    # Prepare font for UI
    font = pygame.font.SysFont(None, 24)
    # UI parameters
    speed_step = 1.0
    dir_step = 15.0
    speed_label_y = 20
    dir_label_y = speed_label_y + 80
    # Initial zoom factor
    zoom = 1.0
    # Initial offset for panning and zoom anchoring
    offset_x, offset_y = 0.0, 0.0

    # Base vegetation surface (unscaled)
    base_veg_surface = pygame.surfarray.make_surface(
        veg_color_map.transpose((1, 0, 2))
    )

    # State variables
    running = True
    sim_running = False
    initial_bc_applied = False
    ignition_map = np.zeros(dem.shape, dtype=np.uint8)

    while running:
        # Compute dynamic layout and scaling
        screen_width, screen_height = screen.get_size()
        map_area_width = screen_width - menu_width
        map_area_height = screen_height
        base_scale = min(map_area_width / W, map_area_height / H)
        cell_size = max(1, int(base_scale * zoom))
        scaled_w = W * cell_size
        scaled_h = H * cell_size
        # Update dynamic UI rectangles
        menu_x = screen_width - menu_width
        menu_rect = pygame.Rect(menu_x, 0, menu_width, screen_height)
        speed_minus_rect = pygame.Rect(menu_x + 10, speed_label_y + 30, 30, 30)
        speed_plus_rect = pygame.Rect(menu_x + 60, speed_label_y + 30, 30, 30)
        dir_minus_rect = pygame.Rect(menu_x + 10, dir_label_y + 30, 30, 30)
        dir_plus_rect = pygame.Rect(menu_x + 60, dir_label_y + 30, 30, 30)
        reset_rect = pygame.Rect(menu_x + 10, dir_label_y + 80, menu_width - 20, 40)
        # Scale the vegetation surface
        scaled_veg_surface = pygame.transform.scale(base_veg_surface, (scaled_w, scaled_h))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resizing
                screen_width, screen_height = event.w, event.h
                screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                # Update menu panel height
                menu_rect.height = screen_height
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    sim_running = not sim_running
                    if sim_running:
                        ign = ignition_map.copy()
                        ignition_map.fill(0)
                        if not initial_bc_applied:
                            bc = PropagatorBoundaryConditions(
                                time=0,
                                ignitions=ign,
                                moisture=moisture,
                                wind_dir=wind_dir,
                                wind_speed=wind_speed,
                            )
                            initial_bc_applied = True
                        else:
                            bc = PropagatorBoundaryConditions(
                                time=simulator.time,
                                ignitions=ign,
                                moisture=None,
                                wind_dir=None,
                                wind_speed=None,
                            )
                        simulator.set_boundary_conditions(bc)
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom in/out around mouse cursor
                mx, my = pygame.mouse.get_pos()
                old_zoom = zoom
                if event.y > 0:
                    zoom = min(zoom * 1.1, 10.0)
                else:
                    zoom = max(zoom * 0.9, 0.1)
                # Adjust offset so the point under the cursor stays in place
                if zoom != old_zoom:
                    ratio = zoom / old_zoom
                    offset_x = mx - (mx - offset_x) * ratio
                    offset_y = my - (my - offset_y) * ratio
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < menu_x:
                    if not sim_running:
                        x = int((mx - offset_x) // cell_size)
                        y = int((my - offset_y) // cell_size)
                        if 0 <= x < W and 0 <= y < H:
                            ignition_map[y, x] ^= 1
                else:
                    # Menu interactions
                    if speed_minus_rect.collidepoint(mx, my):
                        wind_speed_scalar = max(0, wind_speed_scalar - speed_step)
                        wind_speed = np.ones(dem.shape) * wind_speed_scalar
                    elif speed_plus_rect.collidepoint(mx, my):
                        wind_speed_scalar += speed_step
                        wind_speed = np.ones(dem.shape) * wind_speed_scalar
                    elif dir_minus_rect.collidepoint(mx, my):
                        wind_dir_scalar = (wind_dir_scalar - dir_step) % 360
                        wind_dir = np.ones(dem.shape) * wind_dir_scalar
                    elif dir_plus_rect.collidepoint(mx, my):
                        wind_dir_scalar = (wind_dir_scalar + dir_step) % 360
                        wind_dir = np.ones(dem.shape) * wind_dir_scalar
                    elif reset_rect.collidepoint(mx, my):
                        # Reset simulation
                        simulator = Propagator(
                            dem=dem,
                            veg=veg,
                            realizations=100,
                            ros_0=v0,
                            probability_table=prob_table,
                            veg_parameters=p_veg,
                            do_spotting=False,
                            p_time_fn=p_time_wang,
                            p_moist_fn=moist_proba_correction_1,
                        )
                        ignition_map = np.zeros(dem.shape, dtype=np.uint8)
                        initial_bc_applied = False
                        sim_running = False

        # Handle dragging to add ignitions when paused
        if not sim_running and pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            if mx < menu_x:
                x = int((mx - offset_x) // cell_size)
                y = int((my - offset_y) // cell_size)
                if 0 <= x < W and 0 <= y < H:
                    ignition_map[y, x] = 1

        if sim_running:
            nxt_time = simulator.next_time()
            if nxt_time is None:
                sim_running = False
            else:
                simulator.step()

        # Draw vegetation
        # Apply offset for panning/zoom anchoring
        screen.blit(scaled_veg_surface, (int(offset_x), int(offset_y)))

        # Draw fire probability overlay
        if initial_bc_applied:
            proba = simulator.compute_fire_probability()
            proba_map = np.zeros((H, W, 3), dtype=np.uint8)
            proba_map[:, :, 0] = (proba * 255).astype(np.uint8)
            proba_surf = pygame.surfarray.make_surface(
                proba_map.transpose((1, 0, 2))
            )
            proba_surf = pygame.transform.scale(
                proba_surf, (scaled_w, scaled_h)
            )
            proba_surf.set_alpha(128)
            # Apply offset for panning/zoom anchoring
            screen.blit(proba_surf, (int(offset_x), int(offset_y)))

        # Draw ignition points
        ys, xs = np.nonzero(ignition_map)
        for y0, x0 in zip(ys, xs):
            rect = pygame.Rect(
                int(offset_x + x0 * cell_size),
                int(offset_y + y0 * cell_size),
                cell_size, cell_size
            )
            pygame.draw.rect(screen, (255, 255, 0), rect)

        # Draw menu panel and controls
        pygame.draw.rect(screen, (200, 200, 200), menu_rect)
        # Simulation clock
        time_surf = font.render(f"Time: {simulator.time:.1f}", True, (0, 0, 0))
        screen.blit(time_surf, (menu_x + 10, 10))
        # Wind speed controls
        ws_surf = font.render(f"Wind Speed: {wind_speed_scalar:.1f}", True, (0, 0, 0))
        screen.blit(ws_surf, (menu_x + 10, speed_label_y))
        pygame.draw.rect(screen, (180, 180, 180), speed_minus_rect)
        m_surf = font.render("-", True, (0, 0, 0))
        screen.blit(m_surf, (speed_minus_rect.x + (speed_minus_rect.width - m_surf.get_width()) // 2, speed_minus_rect.y + (speed_minus_rect.height - m_surf.get_height()) // 2))
        pygame.draw.rect(screen, (180, 180, 180), speed_plus_rect)
        p_surf = font.render("+", True, (0, 0, 0))
        screen.blit(p_surf, (speed_plus_rect.x + (speed_plus_rect.width - p_surf.get_width()) // 2, speed_plus_rect.y + (speed_plus_rect.height - p_surf.get_height()) // 2))
        # Wind direction controls
        wd_surf = font.render(f"Wind Dir: {wind_dir_scalar:.0f}°", True, (0, 0, 0))
        screen.blit(wd_surf, (menu_x + 10, dir_label_y))
        pygame.draw.rect(screen, (180, 180, 180), dir_minus_rect)
        dm_surf = font.render("-", True, (0, 0, 0))
        screen.blit(dm_surf, (dir_minus_rect.x + (dir_minus_rect.width - dm_surf.get_width()) // 2, dir_minus_rect.y + (dir_minus_rect.height - dm_surf.get_height()) // 2))
        pygame.draw.rect(screen, (180, 180, 180), dir_plus_rect)
        dp_surf = font.render("+", True, (0, 0, 0))
        screen.blit(dp_surf, (dir_plus_rect.x + (dir_plus_rect.width - dp_surf.get_width()) // 2, dir_plus_rect.y + (dir_plus_rect.height - dp_surf.get_height()) // 2))
        # Reset button
        pygame.draw.rect(screen, (255, 100, 100), reset_rect)
        r_surf = font.render("RESET", True, (255, 255, 255))
        screen.blit(r_surf, (reset_rect.x + (reset_rect.width - r_surf.get_width()) // 2, reset_rect.y + (reset_rect.height - r_surf.get_height()) // 2))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()