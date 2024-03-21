# pygame

1. link
   * [documentation/get-started](https://www.pygame.org/wiki/GettingStarted)
   * [documentation/quickstart](https://www.pygame.org/docs/)
   * [youtube-link](https://youtu.be/AY9MnQ4x3zk?si=M3zVLGzGV-xu9xSz) [github-repo](https://github.com/clear-code-projects/UltimatePygameIntro) the ultimate introduction to pygame
   * [github/pygame](https://github.com/pygame/)
   * [github/pygame/example](https://github.com/pygame/pygame/tree/main/examples)
   * [SDL](https://libsdl.org/) Simple DirectMedia Layer
   * [book-link](https://gameprogrammingpatterns.com/contents.html) Game Programming Patterns
2. install
   * `mamba install pygame`
   * `pip install pygame`
   * `python -m pygame.examples.aliens`
3. concept
   * surface
   * screen surface: `pygame.display.set_mode()`
   * HW surface
   * double buffer
   * sprite
   * frame rate: 24 fps for movies
4. pygame feature
   * draft images
   * play sounds
   * check inputs
   * gamedev tools: collision detection, physics, AI, creating text, timer
   * (not pygame) advanced game engines: Unreal, Unity, Godot
5. advice
   * don't bother with pixel-perfect collision detection
   * don't get distracted by side issues
   * event subsystem: state-checking, queue system
   * `poll()` vs `wait()`, `set_blocked()`, `event.get()`, `event.clear()`
   * colorkey blitting, alpha

```Python
pygame.display.set_mode()
pygame.image.load()
pygame.image.load().convert()
pygame.font.Font.render()
pygame.Surface()
Surface.blit()
Surface.fill()
Surface.set_at()
Surface.get_at()
pygame.Rect()

pygame.display.flip()
pygame.display.update()
pygame.sprite.RenderUpdates()

pygame.mouse.get_pos()
pygame.key.get_pressed()

# event type
pygame.QUIT
pygame.ACTIVEEVENT
pygame.KEYDOWN
pygame.KEYUP
pygame.MOUSEMOTION
pygame.MOUSEBUTTONDOWN
pygame.MOUSEBUTTONUP
pygame.JOYAXISMOTION
pygame.JOYBALLMOTION
pygame.JOYHATMOTION
pygame.JOYBUTTONDOWN
pygame.JOYBUTTONUP
pygame.VIDEORESIZE
pygame.VIDEOEXPOSE
pygame.USEREVENT
```
