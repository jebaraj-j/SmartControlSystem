"""
labels.py  —  single source of truth.
Every gesture class, every action, every instruction, every voice intent.
Change here → changes everywhere.
"""

# ─────────────────────────────────────────────────────────
# GESTURE CLASSES  (order = CNN output index)
# IMPORTANT: folder names in dataset/ MUST MATCH these exactly
# ─────────────────────────────────────────────────────────
GESTURE_CLASSES = [
    "R_Cursor",          # 0
    "R_Left_Click",      # 1
    "R_Right_Click",     # 2
    "R_Scroll_Up",       # 3
    "R_Scroll_Down",     # 4
    "R_Open_Palm",       # 5
    "R_Closed_Fist",     # 6
    "L_CTRL_Select",     # 7
    "L_Copy",            # 8
    "L_Cut",             # 9
    "L_Paste",           # 10
    "Both_Shutdown",     # 11
    "R_Tab_Switch",      # 12
    "No_Hand"            # 13
]

NUM_CLASSES = len(GESTURE_CLASSES)   # 14

# ─────────────────────────────────────────────────────────
# What gets printed when a gesture is detected
# ─────────────────────────────────────────────────────────
GESTURE_ACTION = {
    "R_Cursor":        "CURSOR MOVE",
    "R_Left_Click":    "LEFT CLICK (1.5s HOLD)",
    "R_Right_Click":   "RIGHT CLICK (1.5s HOLD)",
    "R_Scroll_Up":     "SCROLL UP",
    "R_Scroll_Down":   "SCROLL DOWN",
    "R_Open_Palm":     "OPEN APPLICATION (5s HOLD)",
    "R_Closed_Fist":   "CLOSE APPLICATION (5s HOLD)",
    "L_CTRL_Select":   "MULTI-SELECTION (CTRL)",
    "L_Copy":          "COPY (1.5s HOLD)",
    "L_Cut":           "CUT (1.5s HOLD)",
    "L_Paste":         "PASTE (1.5s HOLD)",
    "Both_Shutdown":   "SYSTEM SHUTDOWN (CONFIRM)",
    "R_Tab_Switch":    "SWITCH BROWSER TAB",
    "No_Hand":         "IDLE"
}

# ─────────────────────────────────────────────────────────
# Instructions shown during image collection
# ─────────────────────────────────────────────────────────
GESTURE_INSTRUCTIONS = {
    "R_Cursor":
        "RIGHT hand: ONLY index finger up. All other fingers closed.",

    "R_Left_Click":
        "RIGHT hand: Pinch INDEX + THUMB. Hold for 1.5 seconds.",

    "R_Right_Click":
        "RIGHT hand: Pinch MIDDLE + THUMB. Hold for 1.5 seconds.",

    "R_Scroll_Up":
        "RIGHT hand: Index + middle fingers UP (peace sign).",

    "R_Scroll_Down":
        "RIGHT hand: Index + middle fingers DOWN.",

    "R_Open_Palm":
        "RIGHT hand: Fully open palm. Hold steady for 5 seconds.",

    "R_Closed_Fist":
        "RIGHT hand: Closed fist. Hold for 5 seconds.",

    "L_CTRL_Select":
        "LEFT hand: Closed fist (CTRL modifier / multi-select).",

    "L_Copy":
        "LEFT hand: Index finger only. Hold for 1.5 seconds.",

    "L_Cut":
        "LEFT hand: Peace sign (index + middle). Hold for 1.5 seconds.",

    "L_Paste":
        "LEFT hand: Pinky, ring, middle UP. Index + thumb closed.",

    "Both_Shutdown":
        "BOTH hands: Left palm + Right palm closed. Hold 5 seconds.",

    "R_Tab_Switch":
        "RIGHT hand: Index + pinky UP. Hold 2 seconds to activate.",

    "No_Hand":
        "Remove hands completely from camera view."
}

# ─────────────────────────────────────────────────────────
# VOICE COMMAND INTENTS (NO TRAINING REQUIRED)
# Rule-based NLP using keywords
# ─────────────────────────────────────────────────────────
VOICE_COMMANDS = {

    # ── applications ──────────────────────────────────────
    "OPEN_APP": {
        "keywords": ["open", "launch", "start", "run"],
        "action": "OPEN APPLICATION",
        "example": "open chrome",
        "param": "app_name",
    },

    "CLOSE_APP": {
        "keywords": ["close", "exit", "quit", "terminate"],
        "action": "CLOSE APPLICATION",
        "example": "close notepad",
        "param": "app_name",
    },

    "SWITCH_APP": {
        "keywords": ["switch to", "go to", "focus on"],
        "action": "SWITCH APPLICATION",
        "example": "switch to word",
        "param": "app_name",
    },

    # ── window control ────────────────────────────────────
    "MAXIMIZE": {
        "keywords": ["maximize", "full screen"],
        "action": "MAXIMIZE WINDOW",
        "example": "maximize window",
        "param": None,
    },

    "MINIMIZE": {
        "keywords": ["minimize", "hide window"],
        "action": "MINIMIZE WINDOW",
        "example": "minimize chrome",
        "param": None,
    },

    # ── file system ───────────────────────────────────────
    "CREATE_FILE": {
        "keywords": ["create file", "new file"],
        "action": "CREATE FILE",
        "example": "create file report.txt",
        "param": "file_name",
    },

    "CREATE_FOLDER": {
        "keywords": ["create folder", "new folder"],
        "action": "CREATE FOLDER",
        "example": "create folder documents",
        "param": "folder_name",
    },

    "DELETE_FILE": {
        "keywords": ["delete file", "remove file"],
        "action": "DELETE FILE",
        "example": "delete file old.txt",
        "param": "file_name",
    },

    # ── clipboard ─────────────────────────────────────────
    "COPY": {
        "keywords": ["copy"],
        "action": "COPY",
        "example": "copy report.pdf",
        "param": "item_name",
    },

    "CUT": {
        "keywords": ["cut"],
        "action": "CUT",
        "example": "cut image.png",
        "param": "item_name",
    },

    "PASTE": {
        "keywords": ["paste"],
        "action": "PASTE",
        "example": "paste",
        "param": None,
    },

    # ── navigation ────────────────────────────────────────
    "NAVIGATE_BACK": {
        "keywords": ["go back", "back"],
        "action": "NAVIGATE BACK",
        "example": "go back",
        "param": None,
    },

    "NAVIGATE_FORWARD": {
        "keywords": ["go forward", "next"],
        "action": "NAVIGATE FORWARD",
        "example": "next",
        "param": None,
    },

    # ── scroll ────────────────────────────────────────────
    "SCROLL_UP": {
        "keywords": ["scroll up"],
        "action": "SCROLL UP",
        "example": "scroll up",
        "param": None,
    },

    "SCROLL_DOWN": {
        "keywords": ["scroll down"],
        "action": "SCROLL DOWN",
        "example": "scroll down",
        "param": None,
    },
}
