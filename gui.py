# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os

from game_models import GameState, Helping, Player, SYM_TO_ID, ID_TO_SYM, NUM_DICE
from decision_engine import TurnPolicy

class OptimalPlayGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Regenwormen Optimal Play")
        self.geometry("900x800")
        self.minsize(900, 700)  # ensure all buttons fit

        # Load dice images (worm image is the sixth)
        self.dice_images = {}
        image_names = [
            'pictures/dice-six-faces-one.png',
            'pictures/dice-six-faces-two.png',
            'pictures/dice-six-faces-three.png',
            'pictures/dice-six-faces-four.png',
            'pictures/dice-six-faces-five.png',
            'pictures/dice-six-faces-six.png'
        ]
        for i, name in enumerate(image_names):
            if os.path.exists(name):
                img = Image.open(name)
                img = img.resize((50, 50), Image.Resampling.LANCZOS)
                self.dice_images[i] = ImageTk.PhotoImage(img)
            else:
                print(f"Warning: {name} not found, using fallback.")
                self.dice_images[i] = None

        # Turn state
        self.fixed_counts = [0]*6
        self.remaining_dice = []
        self.current_roll_counts = [0]*6

        # Build GUI
        self._build_interface()
        self.reset_turn()

    def _build_interface(self):
        # Main container that expands
        main_frame = ttk.Frame(self)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Top frame: game state inputs (fixed height)
        top_frame = ttk.LabelFrame(main_frame, text="Game State")
        top_frame.pack(fill='x', pady=(0,10))

        # Grill row
        ttk.Label(top_frame, text="Face-up helpings on grill (space‑separated numbers):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.grill_entry = tk.Text(top_frame, height=2, width=50)
        self.grill_entry.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.grill_entry.insert("1.0", " ".join(str(i) for i in range(21, 37)))

        # Opponents row
        ttk.Label(top_frame, text="Opponents' top helpings (leave blank if none):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        opp_frame = ttk.Frame(top_frame)
        opp_frame.grid(row=3, column=0, columnspan=2, pady=5)
        self.opp_entries = []
        for i in range(1, 5):
            ttk.Label(opp_frame, text=f"P{i}:").grid(row=0, column=i-1, padx=5)
            entry = ttk.Entry(opp_frame, width=6)
            entry.grid(row=1, column=i-1, padx=5)
            self.opp_entries.append(entry)

        # Current player top helping
        ttk.Label(top_frame, text="Your top helping (if any, else blank):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.current_top_entry = ttk.Entry(top_frame, width=10)
        self.current_top_entry.grid(row=4, column=1, sticky="w", padx=5)

        top_frame.columnconfigure(0, weight=1)

        # Dice area (expands)
        dice_frame = ttk.LabelFrame(main_frame, text="Dice")
        dice_frame.pack(fill='both', expand=True, pady=(0,10))

        # Use grid inside dice_frame to control expansion
        dice_frame.columnconfigure(0, weight=1)
        dice_frame.rowconfigure(1, weight=1)  # fixed dice canvas row gets extra space
        dice_frame.rowconfigure(3, weight=1)  # remaining dice canvas row gets extra space

        # Fixed dice area
        fixed_label = ttk.Label(dice_frame, text="Fixed dice (taken):")
        fixed_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.fixed_canvas = tk.Canvas(dice_frame, height=60, bg='lightgray')  # reduced height
        self.fixed_canvas.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.fixed_canvas.bind("<Configure>", lambda e: self.update_dice_display())

        # Remaining dice area (clickable)
        remaining_label = ttk.Label(dice_frame, text="Remaining dice (current roll) – click to change face:")
        remaining_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.remaining_canvas = tk.Canvas(dice_frame, height=60, bg='lightgray')  # reduced height
        self.remaining_canvas.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        self.remaining_canvas.bind("<Button-1>", self.on_remaining_dice_click)
        self.remaining_canvas.bind("<Configure>", lambda e: self.update_dice_display())

        # Manual dice addition buttons
        add_frame = ttk.Frame(dice_frame)
        add_frame.grid(row=4, column=0, pady=5, sticky="ew")
        ttk.Label(add_frame, text="Add die to roll:").pack(side="left", padx=5)
        for idx, sym in enumerate(['1','2','3','4','5','Worm']):
            btn = ttk.Button(add_frame, text=sym, command=lambda i=idx: self.add_die(i))
            btn.pack(side="left", padx=2)
        ttk.Button(add_frame, text="Clear roll", command=self.clear_roll).pack(side="left", padx=10)

        # Symbol take buttons
        take_frame = ttk.Frame(dice_frame)
        take_frame.grid(row=5, column=0, pady=10, sticky="ew")
        ttk.Label(take_frame, text="Take all dice of symbol:").pack(side="left", padx=5)
        self.take_buttons = []
        for idx, sym in enumerate(['1','2','3','4','5','Worm']):
            btn = ttk.Button(take_frame, text=sym, command=lambda i=idx: self.take_symbol(i))
            btn.pack(side="left", padx=2)
            self.take_buttons.append(btn)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=5)
        self.stop_btn = ttk.Button(control_frame, text="Stop turn (take tile)", command=self.stop_turn)
        self.stop_btn.pack(side="left", padx=5)
        self.reset_btn = ttk.Button(control_frame, text="Reset turn", command=self.reset_turn)
        self.reset_btn.pack(side="left", padx=5)

        # Decision advice
        advice_frame = ttk.LabelFrame(main_frame, text="Optimal Play Advice")
        advice_frame.pack(fill='x', pady=5)

        self.advice_var = tk.StringVar(value="Click a button for advice")
        advice_label = ttk.Label(advice_frame, textvariable=self.advice_var, font=("Helvetica", 12))
        advice_label.pack(pady=5)

        advice_btn_frame = ttk.Frame(advice_frame)
        advice_btn_frame.pack(pady=5)
        ttk.Button(advice_btn_frame, text="Should I stop now?", command=self.advise_stop).pack(side="left", padx=5)
        ttk.Button(advice_btn_frame, text="Suggest symbol to take", command=self.advise_symbol).pack(side="left", padx=5)

        # Result display
        self.result_var = tk.StringVar(value="")
        result_label = ttk.Label(main_frame, textvariable=self.result_var, font=("Helvetica", 10))
        result_label.pack(pady=5)

    # ---------- Helper methods (unchanged) ----------
    def get_worms(self, number):
        if 21 <= number <= 24:
            return 1
        elif 25 <= number <= 28:
            return 2
        elif 29 <= number <= 32:
            return 3
        elif 33 <= number <= 36:
            return 4
        else:
            return 0

    def get_state_from_input(self):
        grill_input = self.grill_entry.get("1.0", tk.END).strip().split()
        face_up_numbers = set()
        for token in grill_input:
            try:
                num = int(token)
                if 21 <= num <= 36:
                    face_up_numbers.add(num)
            except ValueError:
                continue

        grill = []
        for num in range(21, 37):
            worms = self.get_worms(num)
            h = Helping(num, worms)
            h.face_up = (num in face_up_numbers)
            grill.append(h)

        players = []
        current_player = Player(pid=0)
        current_top = self.current_top_entry.get().strip()
        if current_top.isdigit():
            num = int(current_top)
            if 21 <= num <= 36:
                worms = self.get_worms(num)
                h = Helping(num, worms)
                current_player.add_helping(h)
        players.append(current_player)

        for i, entry in enumerate(self.opp_entries, start=1):
            val_str = entry.get().strip()
            p = Player(pid=i)
            if val_str.isdigit():
                num = int(val_str)
                if 21 <= num <= 36:
                    worms = self.get_worms(num)
                    h = Helping(num, worms)
                    p.add_helping(h)
            players.append(p)

        return GameState(grill=grill, players=players, current_player=0)

    def update_dice_display(self):
        # Fixed canvas
        self.fixed_canvas.delete("all")
        canvas_width = self.fixed_canvas.winfo_width()
        if canvas_width < 10:
            canvas_width = 400
        x, y = 10, 15
        dice_width = 50
        spacing = 5
        step = dice_width + spacing
        for sym_id, count in enumerate(self.fixed_counts):
            for _ in range(count):
                if self.dice_images[sym_id]:
                    self.fixed_canvas.create_image(x, y, image=self.dice_images[sym_id], anchor='nw')
                else:
                    self.fixed_canvas.create_rectangle(x, y, x+40, y+40, fill='white', outline='black')
                    self.fixed_canvas.create_text(x+20, y+20, text=ID_TO_SYM[sym_id])
                x += step
                if x + dice_width > canvas_width:
                    x = 10
                    y += step

        # Remaining canvas
        self.remaining_canvas.delete("all")
        canvas_width = self.remaining_canvas.winfo_width()
        if canvas_width < 10:
            canvas_width = 400
        x, y = 10, 15
        for idx, sym_id in enumerate(self.remaining_dice):
            if self.dice_images[sym_id]:
                self.remaining_canvas.create_image(x, y, image=self.dice_images[sym_id], anchor='nw')
            else:
                self.remaining_canvas.create_rectangle(x, y, x+40, y+40, fill='white', outline='black')
                self.remaining_canvas.create_text(x+20, y+20, text=ID_TO_SYM[sym_id])
            x += step
            if x + dice_width > canvas_width:
                x = 10
                y += step

        self.update_take_buttons()

    def update_take_buttons(self):
        for sym_id in range(6):
            can_take = (self.current_roll_counts[sym_id] > 0) and (self.fixed_counts[sym_id] == 0)
            state = 'normal' if can_take else 'disabled'
            self.take_buttons[sym_id].config(state=state)

    def add_die(self, sym_id):
        total = sum(self.fixed_counts) + len(self.remaining_dice)
        if total >= NUM_DICE:
            messagebox.showinfo("Max dice", f"You already have {NUM_DICE} dice in play.")
            return
        self.remaining_dice.append(sym_id)
        self.current_roll_counts[sym_id] += 1
        self.update_dice_display()

    def clear_roll(self):
        self.remaining_dice = []
        self.current_roll_counts = [0]*6
        self.update_dice_display()

    def on_remaining_dice_click(self, event):
        if not self.remaining_dice:
            return
        x, y = event.x, event.y
        canvas_width = self.remaining_canvas.winfo_width()
        dice_width = 50
        spacing = 5
        step = dice_width + spacing
        start_x, start_y = 10, 15
        temp_x, temp_y = start_x, start_y
        for idx, sym_id in enumerate(self.remaining_dice):
            rect = (temp_x, temp_y, temp_x+dice_width, temp_y+dice_width)
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                new_sym = (sym_id + 1) % 6
                self.current_roll_counts[sym_id] -= 1
                self.current_roll_counts[new_sym] += 1
                self.remaining_dice[idx] = new_sym
                self.update_dice_display()
                return
            temp_x += step
            if temp_x + dice_width > canvas_width:
                temp_x = start_x
                temp_y += step

    def take_symbol(self, sym_id):
        if self.fixed_counts[sym_id] > 0:
            messagebox.showerror("Illegal move", f"You already have {ID_TO_SYM[sym_id]} fixed. Cannot take it again.")
            return
        if self.current_roll_counts[sym_id] == 0:
            return
        count = self.current_roll_counts[sym_id]
        self.fixed_counts[sym_id] += count
        self.remaining_dice = [s for s in self.remaining_dice if s != sym_id]
        self.current_roll_counts[sym_id] = 0
        self.update_dice_display()
        self.advice_var.set(f"Took {count} × {ID_TO_SYM[sym_id]}. You can add more dice or stop.")

    def reset_turn(self):
        self.fixed_counts = [0]*6
        self.remaining_dice = []
        self.current_roll_counts = [0]*6
        self.update_dice_display()
        self.advice_var.set("Add dice to simulate your roll.")
        self.result_var.set("")

    def stop_turn(self):
        try:
            state = self.get_state_from_input()
            collection = tuple(self.fixed_counts)
            from decision_engine import stop_reward
            reward = stop_reward(collection, state)
            if reward > 0:
                outcome = f"Gained {reward} worms"
            elif reward < 0:
                outcome = f"Lost {abs(reward)} worms (failed)"
            else:
                outcome = "No change (failed with no top helping)"
            self.result_var.set(f"Turn ended. {outcome}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def advise_stop(self):
        try:
            state = self.get_state_from_input()
            collection = tuple(self.fixed_counts)
            policy = TurnPolicy(state)
            if policy.should_stop(collection):
                self.advice_var.set("Optimal decision: STOP now")
            else:
                self.advice_var.set("Optimal decision: Continue rolling")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def advise_symbol(self):
        if not self.remaining_dice:
            messagebox.showinfo("No roll", "No dice in the remaining area.")
            return
        try:
            state = self.get_state_from_input()
            collection = tuple(self.fixed_counts)
            roll_counts = tuple(self.current_roll_counts)
            policy = TurnPolicy(state)
            best = policy.choose_symbol(collection, roll_counts)
            if best is None:
                self.advice_var.set("No new symbol available → you will fail if you continue.")
            else:
                self.advice_var.set(f"Best symbol to take: {ID_TO_SYM[best]}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.mainloop()