import pygame

#fontpath = "Im Wunderland.otf"
fontpath = None # use system default
background = (0, 0, 0)
forground  = (255, 255,255)

pygame.init()
screen = pygame.display.set_mode((28, 28))
clock = pygame.time.Clock()
done = False

font = pygame.font.Font(fontpath, 40)

char = 0
text = font.render(str(char), True, forground)
screen.fill(background)
screen.blit(text,(0, 0))

pygame.display.flip()

print("height: ", font.get_height())

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            done = True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
            char += 1
            text = font.render(str(char), True, forground)
            screen.fill(background)
            screen.blit(text,(0, 0))
            pygame.display.flip()


    clock.tick(60)
