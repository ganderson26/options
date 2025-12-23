import pygame.midi

pygame.midi.init()
for i in range(pygame.midi.get_count()):
    r = pygame.midi.get_device_info(i)
    (interface, name, is_input, is_output, opened) = r
    print(f"Device ID: {i}, Name: {name.decode()}, Input: {bool(is_input)}, Output: {bool(is_output)}")
pygame.midi.quit()
