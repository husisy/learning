import pygame


def demo_basic00():
    pygame.init()
    screen = pygame.display.set_mode((320, 240))
    pygame.display.set_caption("demo_basic00")
    clock = pygame.time.Clock()
    surface00 = pygame.Surface((32, 12))
    surface00.fill("red")
    running = True
    while running:
        for event in pygame.event.get(): #poll for events
            if event.type == pygame.QUIT: #clicking the X
                running = False
        screen.fill("purple")
        screen.blit(surface00, (0, 10))
        pygame.display.flip() #put your work on screen
        clock.tick(60)  #limits FPS to 60
    pygame.quit()


def demo_basic01():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("demo_basic01")
    clock = pygame.time.Clock()
    running = True
    dt = 0
    player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill("purple")
        pygame.draw.circle(screen, "red", player_pos, 40)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            player_pos.y -= 300 * dt
        if keys[pygame.K_s]:
            player_pos.y += 300 * dt
        if keys[pygame.K_a]:
            player_pos.x -= 300 * dt
        if keys[pygame.K_d]:
            player_pos.x += 300 * dt
        pygame.display.flip()
        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-independent physics.
        dt = clock.tick(60) / 1000
    pygame.quit()


if __name__ == "__main__":
    demo_basic00()

    # demo_basic01()
