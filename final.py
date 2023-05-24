import cv2
import musx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from musx import Pitch, Interval, gm, MidiFile
from musx.pitch import scale

musx.setmidiplayer("fluidsynth -iq -g2 C:/Users/Matt/fluidsynth-2.3.1-win10-x64/MuseScore_General.sf3 ")

major             = [2, 2, 1, 2, 2, 2, 1]
natural_minor     = [2, 1, 2, 2, 1, 2, 2]
harmonic_minor    = [2, 1, 2, 2, 1, 3, 1]
melodic_minor     = [2, 1, 2, 2, 2, 2, 1]
dorian            = [2, 1, 2, 2, 2, 1, 2]
phrygian          = [1, 2, 2, 2, 1, 2, 2]
lydian            = [2, 2, 2, 1, 2, 2, 1]
mixolydian        = [2, 2, 1, 2, 2, 1, 2]
locrian           = [1, 2, 2, 1, 2, 2, 2]
blues_scale       = [3, 2, 1, 1, 3, 2]
hungarian_minor   = [2, 1, 3, 1, 1, 3, 1]
pentatonic        = [2, 3, 2, 2, 3]
chromatic         = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

SCALES = [major, 
          natural_minor, 
          harmonic_minor, 
          melodic_minor, 
          dorian, 
          phrygian, 
          lydian, 
          mixolydian, 
          locrian, 
          blues_scale, 
          hungarian_minor,
          pentatonic,
          chromatic
         ]

def plotimg(width, height, origRGB, origBGR, hsv):
    dpi = plt.rcParams['figure.dpi']
    figsize = width / float(dpi), height / float(dpi)

    # plot images, where RGB is what the original image is like
    fig, axs = plt.subplots(1, 3, figsize = (20, 20))
    names = ['RGB', 'BGR', 'HSV']
    imgs  = [origRGB, origBGR, hsv]

    for i, x in enumerate(imgs):
        axs[i].title.set_text(names[i])
        axs[i].imshow(x)
        axs[i].grid(False)
    plt.show()

# arrays rescaler using musx.rescale
# default is from 0-255 to 36-96 (C2-C7)
def rescalearr(arr, length=0, minhsv=0, maxhsv=255, tomin=36, tomax=96):
    res = []
    if (length == 0): length = len(arr)
    for i in range(length): # arr
        r = musx.rescale(arr[i], minhsv, maxhsv, tomin, tomax)
        r = round(r, 2) # round to 2 digits
        res.append(r)
        
    return res

# markov analysis of any list
def markovanalysis(arr: list, order: int):
    rules = musx.markov_analyze(arr, order)
    gen = musx.markov(rules)
    return [next(gen) for _ in range(len(arr))]

def plotmarkovimg(markovhsv, width, height, origRGB):
    sortmarkhsv = sorted(markovhsv, key = lambda x: x[0])

    l = len(markovhsv)
    rows = int(np.ceil(l / width))
    cols = int(np.ceil(l / rows))

    # Create a NumPy array of zeros with the desired shape
    hsvnp = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i, hsv in enumerate(markovhsv):
        row = int(i / width)
        col = i % width
        hsvnp[row, col] = hsv
        
    sorthsvnp = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i, hsv in enumerate(sortmarkhsv):
        row = int(i / width)
        col = i % width
        sorthsvnp[row, col] = hsv

    # convert HSV to RGB
    markrgb = cv2.cvtColor(hsvnp, cv2.COLOR_HSV2RGB)
    sortmarkrgb = cv2.cvtColor(sorthsvnp, cv2.COLOR_HSV2RGB)

    # Resize RGBimg, width and height is initialized in the beginning
    hsvimg = cv2.resize(hsvnp, (width, height), interpolation=cv2.INTER_LINEAR)
    rgbimg = cv2.resize(markrgb, (width, height), interpolation=cv2.INTER_LINEAR)
    sortrgbimg = cv2.resize(sortmarkrgb, (width, height), interpolation=cv2.INTER_LINEAR)

    fig, axs = plt.subplots(2, 2, figsize = (20, 20))
    names = ['markov HSVimg', 'markov RGBimg', 'original img', 'sorted markovRGB']
    imgs  = [hsvimg, rgbimg, origRGB, sortrgbimg]

    c=0
    for row in range(2):
        for col in range(2):
            axs[row][col].title.set_text(names[c])
            axs[row][col].imshow(imgs[c])
            c += 1
    plt.show()

# create a key signature and a scale based on the key 
def scalesgen(scalename='major', keysig=None, length=None):
    if (keysig == None):
        keysig = 'A4'
    keypitch = Pitch(keysig)
    if (keypitch.keynum()<36) or (keypitch.keynum()>96):
        raise TypeError(f"'{keysig}' is an invalid pitch.")
        
    scale = []

    if (type(scalename) == list):
        scale = scalename
    else:
        scale = globals()[f"{scalename}"]

    if (length == None): length = len(scale)
    notescale = musx.scale(keypitch.keynum(), length, scale)
    
    printscale = []
    for i in notescale:
        printscale.append(musx.pitch(i).string())
    print(printscale)

    return notescale
    
# creating a melody using percentiles of numbers in hues and a scale  
# repetition (motif) is a list: [whether to contain repeats, amount of notes to repeat, the possibility to repeat, how many notes until a repeat, 
# motif or not (motif is to repeat a segment always from the beginning, while not means to repeat by segments)]
def melwithscale(orighue, notescale, repetition=[True, 10, .5, 20, False]):
    atrep = repetition[1] # the amount of notes to repeat
    repeatconst = 0
    repstart = 0
    repidx = 0
    
    # melody according to the precentile of hues
    num = len(notescale) - 1
    percentiles = np.percentile(orighue, np.linspace(0, 100, num+1))
    
    # melody
    res = []

    for i, h in enumerate(orighue):
        # adding repeats
        if (repetition[0]):                                       # if repeat is true
            if (repeatconst == repetition[3]):                    # if at amount of notes to repeat 
                repeatconst = 0  
                if (random.randint(0, 1) <= repetition[2]):       # if possibile to repeat
                    if (repetition[4]):                           # if repeat a motif
                        res += res[0:repetition[1]]  
                    else:                                         # else not a motif, repeat a segment
                        torep = repetition[1]                     # amount of notes to repeat
                        if (repstart+torep >= len(res)):          # if repeat amount out of bounds
                            res += res[repstart:-1]               # add what is in the melody
                        res += res[repstart:repstart+torep] 
                        repstart = i                              # upload repstart to the new segment
            
        # if hue value falls within the current interval
        for n in range(num):
            if h <= percentiles[n+1]:
                res.append([notescale[n], resats[i], revals[i]])
                break
        # else add add the last note in scale
        else:
            res.append([notescale[-1], resats[i], revals[i]])
        repeatconst += 1
            
    return res

# arguments: the melody to be added harmony, the possibility to add a harmony, one interval, list of intervals, weights of interval, 
# returns a list of harmonies
def addharmony(scalemel, chance=.5, interval='-P5', intervals=None, weights=None, ampli=0.75):
    if (intervals != None) and (weights != None) and (len(intervals) != len(weights)):
        raise TypeError(f"bad intervals and weights lists")
    
    
    if (intervals != None):
        intervals = [Interval(i) for i in intervals]
    else:
        interval = Interval(interval)

    res = []
    for i, mel in enumerate(scalemel):
        if (random.randint(0, 1) > chance):                                  # if not to add a harmony, add a amp 0 note
            res.append([mel[0], mel[1], 0])
            
        else:
            if (intervals != None):                                          # if intervals list is not empty, choose from list
                interval = random.choices(intervals, weights)[0]                
            p = interval.transpose(musx.pitch(mel[0]))
            res.append([p.keynum(), mel[1], mel[2]*ampli])                   # add the harmony with *ampli (default 0.75)
    return res

def playrescale(score, length, h, s, v):
    for i in range(length):
        rano = random.randint(0, 3)
        octaves = [-12, 0, 0, 12] # adding some random octaves to make it more interesting
        
        dur = s[i]
        if (s[i] <= 0):
            dur = .25
        n = musx.Note(time=score.now, duration=dur, pitch=h[i]+octaves[rano], amplitude=v[i]*0.9)
        score.add(n)
        yield (s[i]/2)

def playcomposer(comp, args):
    t=musx.Seq()
    q=musx.Score(out=t)
    args.insert(0, q)
    q.compose( [0, comp(*args)] )
    outputname = image_name.split('.')[0]
    f=musx.MidiFile(f"{outputname}.mid", t).write()
    print(f"{outputname}.mid done")

def playpercent(score, length, hsv, inst=0, a=0.8, ):
    for i in range(length):
        rano = random.randint(0, 3)
        octaves = [-12, 0, 0, 12] # adding some random octaves to make it more interesting
        
        dur = hsv[i][1]
        if (dur <= 0):
            dur = .25
        n = musx.Note(time=score.now, duration=dur, pitch=hsv[i][0]+octaves[rano], amplitude=hsv[i][2]*a, instrument=inst)
        score.add(n)
        yield (hsv[i][1]/2)

if __name__ == '__main__':
    images = ['mypixel.png', 'mymess.png', 'pixelcat.png', 'Xcom2.png', 'gdimg.jpg']
    inimage = input("type in your image name and file extension (ex. picture.jpg), if you do not have one type \"none\"")
    print(inimage)
    
    image_name = images[-1]
    if (inimage != "none"):
        image_name = inimage

    origBGR = cv2.imread(image_name)

    # RGB image
    origRGB = cv2.cvtColor(origBGR, cv2.COLOR_BGR2RGB)
    # hsv image
    hsv = cv2.cvtColor(origBGR, cv2.COLOR_BGR2HSV)

    # dimensions of image
    height, width, depth = hsv.shape
    print("dimensions of img ", height, width)

    # plot image

    # if use random pixel
    inranpxl = input("do you want to use random pixels? type 'yes' or 'no'")
    randomPxl = False
    if (inranpxl == "yes"):
        randomPxl = True

    h = 0
    w = 0
    # HSV = Hues, Saturation, Value(Brightness) -> pitch, rhy, amp
    # The original HSV values of the image
    hues = []
    sats = []
    vals = []

    for h in range(height):
        for w in range(width):
            y = h
            x = w
            if (randomPxl):
                y = random.randint(0, height-1)
                x = random.randint(0, width-1)
            # Values at pixel coordinate (w, h)
            hue = hsv[y][x][0] 
            sat = hsv[y][x][1]
            val = hsv[y][x][2]
            hues.append(hue)
            sats.append(sat)
            vals.append(val)
            
    # print(max(hues), min(hues))
    # print("--------------------------")
    # for i in range(10):
    #     print(hues[i], sats[i], vals[i])
        
    print("hsv arr done")

    # rescaled HSV, where 180 is the max hue rescaled to Midi# 36-96, and SV are rescaled to 0-1 for rhythm and amplitude
    maxhue = 180
    rehues = rescalearr(hues, maxhsv=maxhue)
    resats = rescalearr(sats, tomin=0, tomax=1)
    revals = rescalearr(vals, tomin=0, tomax=1)

    # for i in range(200):
    #     print(rehues[i], resats[i], revals[i])
        
    print("rescaled hsv done")

    now = 200
    # what it sounds like so far, using only 50 notes
    playcomposer(playrescale, [now, rehues, resats, revals])

    # print(rehues[0:now])
    # print(resats[0:now])
    # print(revals[0:now])
    print('playing rescale!')

    # markov analysis on rescaled hsv
    order = 1
    markhue = markovanalysis(rehues, order)
    marksat = markovanalysis(resats, order)
    markval = markovanalysis(revals, order)

    # rescale them back to HSV format to output an image
    remhue = rescalearr(markhue, minhsv=36, maxhsv=96, tomin=0, tomax=maxhue)
    remsat = rescalearr(marksat, maxhsv=1, tomin=0, tomax=255)
    remval = rescalearr(marksat, maxhsv=1, tomin=0, tomax=255)

    markovhsv = list(zip(remhue, remsat, remval))

    # plotmarkovimg(markovhsv, width, height, origRGB)

    # How to add tonality to music by Chatgpt:
    # To add tonality or musicality to a list of chromatic random notes, you could try the following:
    # 
    # Identify a key: Choose a key that fits the notes you have and try to use the notes of that key as a foundation for your melody. This will give your melody a tonal center and create a sense of coherence.
    # 
    # Add harmony: Once you have a melody, try adding chords that fit the key you've chosen. This will add depth and richness to your melody and create a sense of harmony.
    # 
    # Use repetition: Repetition can help to reinforce the tonality of your melody and create a sense of structure. Try repeating certain notes or phrases throughout your melody.


    inscale = input("to make the music tonal, do you want to define your own scale?, type 'yes' or 'no'")
    if (inscale == "yes"):
        inscale = input("please define your own scales separated with commas (ex. 2,1,2,3)")
        inscale = inscale.split(",")
        inscale = [int(i) for i in inscale]
    else:
        inscale = input("""please select a scale from this 
            (major, 
            natural_minor, 
            harmonic_minor, 
            melodic_minor, 
            dorian, 
            phrygian, 
            lydian, 
            mixolydian, 
            locrian, 
            blues_scale, 
            hungarian_minor,
            pentatonic,
            chromatic)""")
    inlength = input("please define the length of your scale, type 'no' if you do not want to")
    if inlength == "no":
        inlength = None
    inkeysig = input("please define a key signature (ex. C#4, Ab3), type 'no' if you do not want to and it will default to A4")
    if inkeysig == "no":
        inkeysig = "A4"

    # create an A major scale
    # amajor = scalesgen(scalename='major', keysig='A4')
    # csharminor = scalesgen( scalename='harmonic_minor', keysig='C#4')
    # myscale = scalesgen(scalename=[2, 2, 1, 2, 5], keysig='C4')
    definedscale = scalesgen(inscale, inkeysig, inlength)

    # now i have a tonal center or a scale i want to work on, I then generate a melody using the hues and the scale
    # rhy and amp will still use the rescaled values (resats, revals)

    # print(hues[0:200])
    mws = melwithscale(hues[0:200], definedscale)
    # print(len(mws))
    # print(mws[0:10])
    # print(mws[0:500])
    print("tonal melody done")

        # add a harmony
    inchance = input("what is the chance you want harmonies to occur? type 'no' will default it to 0.6")
    if (inchance == 'no'):
        inchance = .6

    ininterval = input("do you want 'one' or 'many' harmonies?")
    inintervals = None
    inweight = None
    if (ininterval == 'many'):
        inintervals = input("please type in your intervals of harmony separated by commas '+' their weights (ex. P1,-P4,P5+0.5,0.5,0.5)")
        inintervals = inintervals.split("+")
        inweight = inintervals[1].split(',')
        inweight = [float(i) for i in inweight]
        inintervals = inintervals[0].split(',')
        
    else:
        ininterval = input("please type in your interval of harmony (ex. -P8 means an octave lower)")

    # adding a harmony with P1 and P4 lower: P1,-P4+0.5,0.5
    # ah = addharmony(mws, chance=.7, intervals=['P1', '-P4'], weights=[0.5, 0.5])
    # # adding a bass with an octave lower
    # bah = addharmony(mws, chance=.4, interval='-P8')
    userh = addharmony(mws, inchance, ininterval, inintervals, inweight)
    print("harmony!")
            
    num = 200

    t=musx.Seq()
    instruments = {0: gm.BrightAcousticPiano, 1: gm.Marimba, 2: gm.Flute, 3: gm.Woodblock}
    q=musx.Score(out=t)
    meta = MidiFile.metatrack( ins = instruments )
    q.compose([
            [0, playpercent(q, num, mws, 0)],
            # [0, playpercent(q, num, ah, 1, 0.6)],
            # [0, playpercent(q, num, bah, 1)],
            [0, playpercent(q, num, userh, 1, 0.6)]
            ])

    outputname = "tonal" + image_name.split('.')[0]
    f=musx.MidiFile(f"{outputname}.mid", [meta, t]).write()
    print(f"{outputname}.mid with tonality done")