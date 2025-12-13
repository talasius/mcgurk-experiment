#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on December 09, 2025, at 18:26
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'mcgurk'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\aivan\\OneDrive\\Documents\\University\\Cognitive Psychology\\experiment\\mcgurk.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='laptop17Inch', color=[1.0000, 0.7882, 0.7647], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 0.7882, 0.7647]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('proceedButton') is None:
        # initialise proceedButton
        proceedButton = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='proceedButton',
        )
    if deviceManager.getDevice('submitButton') is None:
        # initialise submitButton
        submitButton = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='submitButton',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # enter 'rush' mode (raise CPU priority)
    if not PILOTING:
        core.rush(enable=True)
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    # Run 'Begin Experiment' code from prepareSequence
    
    
    instructionsText = visual.TextStim(win=win, name='instructionsText',
        text='Добро пожаловать!\n\nВы увидите несколько коротких видео\nПосле просмотра видео, вам будет предложено ответить, что вы услышали\n\n Если готовы, то нажмите пробел для того, чтобы начать',
        font='Courier New',
        units='norm', pos=(0, 0), draggable=False, height=0.085, wrapWidth=1.8, ori=0.0, 
        color=[-0.8902, -0.7490, -0.8039], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    proceedButton = keyboard.Keyboard(deviceName='proceedButton')
    # Run 'Begin Experiment' code from textOptions
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    instructionsText.alignText= 'center'
    
    
    # --- Initialize components for Routine "fixation" ---
    fixationScreen = visual.TextStim(win=win, name='fixationScreen',
        text='*',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color=[-0.8902, -0.7490, -0.8039], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial" ---
    # Run 'Begin Experiment' code from prepareTrial
    import random
    from psychopy import data
    congurentVideo = visual.MovieStim(
        win, name='congurentVideo',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=(1.8, 1.0), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=-1
    )
    
    # --- Initialize components for Routine "response" ---
    responseScreen = visual.TextStim(win=win, name='responseScreen',
        text='Что вы услышали?',
        font='Courier New',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-0.8902, -0.7490, -0.8039], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    responseForm = visual.TextBox2(
         win, text=None, placeholder='Напишите, что вы услышали', font='Courier New',
         ori=0.0, pos=(0, 0), draggable=False, units='norm',     letterHeight=0.05,
         size=(0.9, 0.1), borderWidth=1.5,
         color=[-0.8902, -0.7490, -0.8039], colorSpace='rgb',
         opacity=1.0,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.01, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-0.8196, -0.5059, -0.5529],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='responseForm',
         depth=-1, autoLog=True,
    )
    submitButton = keyboard.Keyboard(deviceName='submitButton')
    isiScreen = visual.TextStim(win=win, name='isiScreen',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "thank_you" ---
    thankYouText = visual.TextStim(win=win, name='thankYouText',
        text='Спасибо за участие!',
        font='Courier New',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color=[-0.8902, -0.7490, -0.8039], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[instructionsText, proceedButton],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from prepareSequence
    import random
    from psychopy import prefs
    # create starting attributes for proceedButton
    proceedButton.keys = []
    proceedButton.rt = []
    _proceedButton_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructionsText* updates
        
        # if instructionsText is starting this frame...
        if instructionsText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructionsText.frameNStart = frameN  # exact frame index
            instructionsText.tStart = t  # local t and not account for scr refresh
            instructionsText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructionsText, 'tStartRefresh')  # time at next scr refresh
            # update status
            instructionsText.status = STARTED
            instructionsText.setAutoDraw(True)
        
        # if instructionsText is active this frame...
        if instructionsText.status == STARTED:
            # update params
            pass
        
        # *proceedButton* updates
        waitOnFlip = False
        
        # if proceedButton is starting this frame...
        if proceedButton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            proceedButton.frameNStart = frameN  # exact frame index
            proceedButton.tStart = t  # local t and not account for scr refresh
            proceedButton.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(proceedButton, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'proceedButton.started')
            # update status
            proceedButton.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(proceedButton.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(proceedButton.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if proceedButton.status == STARTED and not waitOnFlip:
            theseKeys = proceedButton.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _proceedButton_allKeys.extend(theseKeys)
            if len(_proceedButton_allKeys):
                proceedButton.keys = _proceedButton_allKeys[0].name  # just the first key pressed
                proceedButton.rt = _proceedButton_allKeys[0].rt
                proceedButton.duration = _proceedButton_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if proceedButton.keys in ['', [], None]:  # No response was made
        proceedButton.keys = None
    thisExp.addData('proceedButton.keys',proceedButton.keys)
    if proceedButton.keys != None:  # we had a response
        thisExp.addData('proceedButton.rt', proceedButton.rt)
        thisExp.addData('proceedButton.duration', proceedButton.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks_loop = data.TrialHandler2(
        name='blocks_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(blocks_loop)  # add the loop to the experiment
    thisBlocks_loop = blocks_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlocks_loop.rgb)
    if thisBlocks_loop != None:
        for paramName in thisBlocks_loop:
            globals()[paramName] = thisBlocks_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlocks_loop in blocks_loop:
        blocks_loop.status = STARTED
        if hasattr(thisBlocks_loop, 'status'):
            thisBlocks_loop.status = STARTED
        currentLoop = blocks_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlocks_loop.rgb)
        if thisBlocks_loop != None:
            for paramName in thisBlocks_loop:
                globals()[paramName] = thisBlocks_loop[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[fixationScreen],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = None
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisBlocks_loop, 'status') and thisBlocks_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixationScreen* updates
            
            # if fixationScreen is starting this frame...
            if fixationScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationScreen.frameNStart = frameN  # exact frame index
                fixationScreen.tStart = t  # local t and not account for scr refresh
                fixationScreen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationScreen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixationScreen.started')
                # update status
                fixationScreen.status = STARTED
                fixationScreen.setAutoDraw(True)
            
            # if fixationScreen is active this frame...
            if fixationScreen.status == STARTED:
                # update params
                pass
            
            # if fixationScreen is stopping this frame...
            if fixationScreen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationScreen.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixationScreen.tStop = t  # not accounting for scr refresh
                    fixationScreen.tStopRefresh = tThisFlipGlobal  # on global time
                    fixationScreen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationScreen.stopped')
                    # update status
                    fixationScreen.status = FINISHED
                    fixationScreen.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[congurentVideo],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        congurentVideo.setMovie(video_file)
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlocks_loop, 'status') and thisBlocks_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *congurentVideo* updates
            
            # if congurentVideo is starting this frame...
            if congurentVideo.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                congurentVideo.frameNStart = frameN  # exact frame index
                congurentVideo.tStart = t  # local t and not account for scr refresh
                congurentVideo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(congurentVideo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'congurentVideo.started')
                # update status
                congurentVideo.status = STARTED
                congurentVideo.setAutoDraw(True)
                congurentVideo.play()
            
            # if congurentVideo is stopping this frame...
            if congurentVideo.status == STARTED:
                if bool(False) or congurentVideo.isFinished:
                    # keep track of stop time/frame for later
                    congurentVideo.tStop = t  # not accounting for scr refresh
                    congurentVideo.tStopRefresh = tThisFlipGlobal  # on global time
                    congurentVideo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'congurentVideo.stopped')
                    # update status
                    congurentVideo.status = FINISHED
                    congurentVideo.setAutoDraw(False)
                    congurentVideo.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        congurentVideo.stop()  # ensure movie has stopped at end of Routine
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "response" ---
        # create an object to store info about Routine response
        response = data.Routine(
            name='response',
            components=[responseScreen, responseForm, submitButton, isiScreen],
        )
        response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        responseForm.reset()
        # create starting attributes for submitButton
        submitButton.keys = []
        submitButton.rt = []
        _submitButton_allKeys = []
        # store start times for response
        response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        response.tStart = globalClock.getTime(format='float')
        response.status = STARTED
        thisExp.addData('response.started', response.tStart)
        response.maxDuration = None
        # keep track of which components have finished
        responseComponents = response.components
        for thisComponent in response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "response" ---
        response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlocks_loop, 'status') and thisBlocks_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *responseScreen* updates
            
            # if responseScreen is starting this frame...
            if responseScreen.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                responseScreen.frameNStart = frameN  # exact frame index
                responseScreen.tStart = t  # local t and not account for scr refresh
                responseScreen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(responseScreen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'responseScreen.started')
                # update status
                responseScreen.status = STARTED
                responseScreen.setAutoDraw(True)
            
            # if responseScreen is active this frame...
            if responseScreen.status == STARTED:
                # update params
                pass
            
            # if responseScreen is stopping this frame...
            if responseScreen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > responseScreen.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    responseScreen.tStop = t  # not accounting for scr refresh
                    responseScreen.tStopRefresh = tThisFlipGlobal  # on global time
                    responseScreen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'responseScreen.stopped')
                    # update status
                    responseScreen.status = FINISHED
                    responseScreen.setAutoDraw(False)
            
            # *responseForm* updates
            
            # if responseForm is starting this frame...
            if responseForm.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                responseForm.frameNStart = frameN  # exact frame index
                responseForm.tStart = t  # local t and not account for scr refresh
                responseForm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(responseForm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'responseForm.started')
                # update status
                responseForm.status = STARTED
                responseForm.setAutoDraw(True)
            
            # if responseForm is active this frame...
            if responseForm.status == STARTED:
                # update params
                pass
            
            # *submitButton* updates
            waitOnFlip = False
            
            # if submitButton is starting this frame...
            if submitButton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                submitButton.frameNStart = frameN  # exact frame index
                submitButton.tStart = t  # local t and not account for scr refresh
                submitButton.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(submitButton, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'submitButton.started')
                # update status
                submitButton.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(submitButton.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(submitButton.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if submitButton.status == STARTED and not waitOnFlip:
                theseKeys = submitButton.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                _submitButton_allKeys.extend(theseKeys)
                if len(_submitButton_allKeys):
                    submitButton.keys = _submitButton_allKeys[-1].name  # just the last key pressed
                    submitButton.rt = _submitButton_allKeys[-1].rt
                    submitButton.duration = _submitButton_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *isiScreen* updates
            
            # if isiScreen is starting this frame...
            if isiScreen.status == NOT_STARTED and tThisFlip >= submitButton.status == FINISHED-frameTolerance:
                # keep track of start time/frame for later
                isiScreen.frameNStart = frameN  # exact frame index
                isiScreen.tStart = t  # local t and not account for scr refresh
                isiScreen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(isiScreen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'isiScreen.started')
                # update status
                isiScreen.status = STARTED
                isiScreen.setAutoDraw(True)
            
            # if isiScreen is active this frame...
            if isiScreen.status == STARTED:
                # update params
                pass
            
            # if isiScreen is stopping this frame...
            if isiScreen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > isiScreen.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    isiScreen.tStop = t  # not accounting for scr refresh
                    isiScreen.tStopRefresh = tThisFlipGlobal  # on global time
                    isiScreen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'isiScreen.stopped')
                    # update status
                    isiScreen.status = FINISHED
                    isiScreen.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=response,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "response" ---
        for thisComponent in response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for response
        response.tStop = globalClock.getTime(format='float')
        response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('response.stopped', response.tStop)
        blocks_loop.addData('responseForm.text',responseForm.text)
        # check responses
        if submitButton.keys in ['', [], None]:  # No response was made
            submitButton.keys = None
        blocks_loop.addData('submitButton.keys',submitButton.keys)
        if submitButton.keys != None:  # we had a response
            blocks_loop.addData('submitButton.rt', submitButton.rt)
            blocks_loop.addData('submitButton.duration', submitButton.duration)
        # the Routine "response" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisBlocks_loop as finished
        if hasattr(thisBlocks_loop, 'status'):
            thisBlocks_loop.status = FINISHED
        # if awaiting a pause, pause now
        if blocks_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            blocks_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'blocks_loop'
    blocks_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "thank_you" ---
    # create an object to store info about Routine thank_you
    thank_you = data.Routine(
        name='thank_you',
        components=[thankYouText],
    )
    thank_you.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for thank_you
    thank_you.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thank_you.tStart = globalClock.getTime(format='float')
    thank_you.status = STARTED
    thisExp.addData('thank_you.started', thank_you.tStart)
    thank_you.maxDuration = None
    # keep track of which components have finished
    thank_youComponents = thank_you.components
    for thisComponent in thank_you.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thank_you" ---
    thank_you.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thankYouText* updates
        
        # if thankYouText is starting this frame...
        if thankYouText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thankYouText.frameNStart = frameN  # exact frame index
            thankYouText.tStart = t  # local t and not account for scr refresh
            thankYouText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thankYouText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thankYouText.started')
            # update status
            thankYouText.status = STARTED
            thankYouText.setAutoDraw(True)
        
        # if thankYouText is active this frame...
        if thankYouText.status == STARTED:
            # update params
            pass
        
        # if thankYouText is stopping this frame...
        if thankYouText.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thankYouText.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                thankYouText.tStop = t  # not accounting for scr refresh
                thankYouText.tStopRefresh = tThisFlipGlobal  # on global time
                thankYouText.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thankYouText.stopped')
                # update status
                thankYouText.status = FINISHED
                thankYouText.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=thank_you,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            thank_you.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thank_you.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thank_you" ---
    for thisComponent in thank_you.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thank_you
    thank_you.tStop = globalClock.getTime(format='float')
    thank_you.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thank_you.stopped', thank_you.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thank_you.maxDurationReached:
        routineTimer.addTime(-thank_you.maxDuration)
    elif thank_you.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)
    # end 'rush' mode
    core.rush(enable=False)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
