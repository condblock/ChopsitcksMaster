import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox

# ----------------------------------------------------------------
# atk, move function
#
# key는 l 또는 r
def atk(atkdict, atk_key, victdict, vict_key): # atkdict[atk_key] > 0
    victdict[vict_key] += atkdict[atk_key]
    if victdict[vict_key] > 5:
        victdict[vict_key] = 0

def move(dict, key, amount): # dict[key] > 0, result of dict[key] > 0, result of dict[other] <= 5
    if key == 'l':
        dict['l'] += amount
        dict['r'] -= amount
    elif key == 'r':
        dict['r'] += amount
        dict['l'] -= amount
    # sub key1 and add it to key2

def check(): # check win or lose
    global p1
    global p2
    if p1['l'] == 0 and p1['r'] == 0:
        tkinter.messagebox.showinfo(title='패배!', message='패배했습니다! 진행 상황을 초기화합니다.')
        p1 = {'l': 1, 'r': 1}
        p2 = {'l': 1, 'r': 1}
    elif p2['l'] == 0 and p2['r'] == 0:
        tkinter.messagebox.showinfo(title='승리!', message='승리했습니다! 진행 상황을 초기화합니다.')
        p1 = {'l': 1, 'r': 1}
        p2 = {'l': 1, 'r': 1}
    # true means lose
#
# ----------------------------------------------------------------
# GUI 관련 함수

p1 = {'l': 1, 'r': 3}
p2 = {'l': 1, 'r': 1}

# GUI 설정

class Atk_gui(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.selected_p1 = tk.StringVar()
        self.selected_p2 = tk.StringVar()
        self.geometry('300x250')
        self.title('공격')
        ttk.Label(self, text="공격할 손을 선택하세요").pack(expand=True)

        ttk.Label(self, text="내 손:").pack(expand=True)
        self.selected_p1 = tk.StringVar()
        r1 = ttk.Radiobutton(self, text='왼손(현재 {}개)'.format(p1['l']), value='l', variable=self.selected_p1)
        r1.pack(fill='x', padx=5, pady=5)
        r2 = ttk.Radiobutton(self, text='오른손(현재 {}개)'.format(p1['r']), value='r', variable=self.selected_p1)
        r2.pack(fill='x', padx=5, pady=5)

        ttk.Label(self, text="상대 손:").pack(expand=True)
        self.selected_p2 = tk.StringVar()
        r1 = ttk.Radiobutton(self, text='왼손(현재 {}개)'.format(p2['l']), value='l', variable=self.selected_p2)
        r1.pack(fill='x', padx=5, pady=5)
        r2 = ttk.Radiobutton(self, text='오른손(현재 {}개)'.format(p2['r']), value='r', variable=self.selected_p2)
        r2.pack(fill='x', padx=5, pady=5)

        ttk.Button(self,
                text='확인',
                command=self.sendresult).pack(expand=True)
        ttk.Button(self,
                text='취소',
                command=self.destroy).pack(expand=True)
    def sendresult(self):
        global p1
        global p2
        sel_p1 = self.selected_p1.get()
        sel_p2 = self.selected_p2.get()
        print(sel_p1, sel_p2)
        if p1[sel_p1] == 0:
            tkinter.messagebox.showerror(title='오류', message='0인 손으로 공격할 수 없습니다.')
        elif p2[sel_p2] == 0:
            tkinter.messagebox.showerror(title='오류', message='0인 손을 공격할 수 없습니다.')
        else:
            atk(p1, sel_p1, p2, sel_p2)
            tkinter.messagebox.showinfo(title='공격', message='공격에 성공했습니다!\n\n현재 상황:\n내 손: 왼손 {0}, 오른손 {1}\n상대 손: 왼손 {2}, 오른손 {3}'
                                        .format(p1['l'], p1['r'], p2['l'], p2['r']))
            check()
            self.destroy()
        self.destroy()
class Move_gui(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.geometry('300x200')
        self.title('이동')
        ttk.Label(self, text="선택하지 않은 손에서 빼서 선택한 손에 더합니다").pack(expand=True)

        ttk.Label(self, text="선택할 손:").pack(expand=True)
        self.selected_p1 = tk.StringVar()
        r1 = ttk.Radiobutton(self, text='왼손(현재 {}개)'.format(p1['l']), value='l', variable=self.selected_p1)
        r1.pack(fill='x', padx=5, pady=5)
        r2 = ttk.Radiobutton(self, text='오른손(현재 {}개)'.format(p1['r']), value='r', variable=self.selected_p1)
        r2.pack(fill='x', padx=5, pady=5)

        self.value = tk.StringVar()
        spin_box = ttk.Spinbox(
            self,
            from_=0,
            to=4,
            values=(1, 2, 3, 4, 5),
            textvariable=self.value,
            wrap=True)
        spin_box.pack()

        ttk.Button(self,
                text='확인',
                command=self.sendresult).pack(expand=True)
        ttk.Button(self,
                text='취소',
                command=self.destroy).pack(expand=True)
    def sendresult(self):
        global p1
        global p2
        sel_p1 = self.selected_p1.get()
        if sel_p1 == 'l':
            sel_p2 = 'r'
        elif sel_p1 == 'r':
            sel_p2 = 'l'
        value = int(self.value.get())
        print(sel_p1, self.value)
        print((p1[sel_p1], p1[sel_p2]), (p1[sel_p2] - value, p1[sel_p1] + value))
        if value >= 1 and value <= p1[sel_p2]:
            move(p1, sel_p1, value)
            tkinter.messagebox.showinfo(title='이동', message='이동에 성공했습니다!\n\n현재 상황:\n내 손: 왼손 {0}, 오른손 {1}\n상대 손: 왼손 {2}, 오른손 {3}'
                                        .format(p1['l'], p1['r'], p2['l'], p2['r']))
            self.destroy()
        elif p1[sel_p1] + value > 5:
            tkinter.messagebox.showerror(title='오류', message='선택한 값이 너무 커서 더할 수 없습니다.')
        elif p1[sel_p2] - value < 0:
            tkinter.messagebox.showerror(title='오류', message='선택한 값이 너무 커서 뺼 수 없습니다.')
        elif (p1[sel_p1], p1[sel_p2]) == (p1[sel_p2] - value, p1[sel_p1] + value):
            tkinter.messagebox.showerror(title='오류', message='값의 위치만 바꿀 수 없습니다.')
        self.destroy()
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry('300x200')
        self.title('젓가락 AI')

        # place a button on the root window
        ttk.Button(self,
                text='공격하기',
                command=self.open_atk).pack(expand=True)
        ttk.Button(self,
                text='이동하기',
                command=self.open_move).pack(expand=True)

    def open_atk(self):
        atk_gui = Atk_gui(self)
        Atk_gui.grab_set(self)

    def open_move(self):
        move_gui = Move_gui(self)
        Move_gui.grab_set(self)
        


if __name__ == "__main__":
    app = App()
    app.mainloop()