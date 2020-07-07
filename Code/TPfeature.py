import numpy as np
class TPfeature():
    def __init__(self,
                 gchHoriWidth = None,
                 gchVertWidth = None,
                 guc2DChannelNum = None,
                 gucAddStageNum = None,
                 gchHoriStart = None,
                 gchHoriEnd = None,
                 gchVertStart = None,
                 gchVertEnd = None,
                 gchCinScale = None,
                 gchCinScaleLevel = None,
                 gucMarginLeft = None,
                 gucMarginRight = None,
                 gucMarginUp = None,
                 gucMarginDown = None,
                 gwBoundaryX = None,
                 gwBoundaryY = None,
                 gwScreenX = None,
                 gwScreenY = None,
                 gwXResolution = None,
                 gwYResolution = None):
        ### Original define ####
        self.gchHoriWidth = gchHoriWidth
        self.gchVertWidth = gchVertWidth

        if (guc2DChannelNum!=None):
            self.guc2DChannelNum = guc2DChannelNum
        self.gucAddStageNum = gucAddStageNum
        
        if gucAddStageNum is None:
            self.AllStageNumber = guc2DChannelNum
        else:
            self.AllStageNumber = guc2DChannelNum + gucAddStageNum

        self.gchHoriStart = gchHoriStart
        self.gchHoriEnd = gchHoriEnd
        self.gchVertStart = gchVertStart
        self.gchVertEnd = gchVertEnd
        self.gchCinScale = gchCinScale
        self.gchCinScaleLevel = gchCinScaleLevel
        self.gucMarginLeft = gucMarginLeft
        self.gucMarginRight = gucMarginRight
        self.gucMarginUp = gucMarginUp
        self.gucMarginDown = gucMarginDown
        
        if gwXResolution is None:
            self.gwXResolution = gchHoriWidth * gchCinScale
        else:
            self.gwXResolution = gwXResolution
        if gwYResolution is None:
            self.gwYResolution = gchVertWidth * gchCinScale
        else:
            self.gwYResolution = gwYResolution

        if gwScreenX is None:
            self.gwScreenX = self.gwXResolution
        else:  
            self.gwScreenX = gwScreenX
        
        if gwScreenY is None:
            self.gwScreenY = self.gwYResolution
        else:
            self.gwScreenY = gwScreenY

        if gwBoundaryX is None:
            self.gwBoundaryX = self.gwScreenX
        else:    
            self.gwBoundaryX = gwBoundaryX

        if gwBoundaryY is None:
            self.gwBoundaryY = self.gwScreenX
        else:
            self.gwBoundaryY = gwBoundaryY
        ## Call the Jason Define ####
        self.JasonDefine()

    def JasonDefine(self, Offest = None):
        self.Offset = Offest
        

