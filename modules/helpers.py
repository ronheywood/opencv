import random
def get_random_test_image():
    images = [
            'at-rest-teed-up.png',
            'at-rest.png',
            'ball-in-hand.jpg',
            'close-up-outdoor--no-club-grass.png',
            #'close-up-outdoor-address.png', #This one seems to be missed by the detection rules
            'close-up-outdoor-grass-impact.png',
            'driver-cimpression.png',
            'driver-launch.png',
            'high-spin-lob-wedge.png',
            'low-loft-high-speed-iron.png',
            'punch-low-launch.png',
            'putter-launch.png']
    index = random.randint(0,len(images)-1)
    return f'test-images/{images[index]}'