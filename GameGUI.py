import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import numpy as np

# 공격, 이동 함수
def atk(state, atk, vict):
    state[1, vict] += state[0, atk]
    if state[1, vict] > 4:
        state[1, vict] = 0
    return state

def move(state, value):
    new_state = state.copy()
    new_state[0, 0] += value
    new_state[0, 1] -= value
    return new_state

# Q값 불러오기
Q = np.load('C:/Users/user/Documents/GitHub/ChopsitcksMaster/Q.npy')

# 상태를 숫자로 변환
def get_state_number(state):
    return state[0, 0] * 1 + state[0, 1] * 5 + state[1, 0] * 25 + state[1, 1] * 125


# ai 행동
def ai_action(state):
    global Q
    state_temp = state[::-1]

    act_number = get_state_number(state_temp)
    ai_action = np.argmax(Q[act_number, :])
    print(Q[act_number, :])
    action_str = ""
    if ai_action == 0:
        state_temp = atk(state_temp, 0, 0)
        action_str = f"AI가 왼손으로 왼손을 공격했습니다."
    elif ai_action == 1:
        state_temp = atk(state_temp, 0, 1)
        action_str = f"AI가 왼손으로 오른손을 공격했습니다."
    elif ai_action == 2:
        state_temp = atk(state_temp, 1, 0)
        action_str = f"AI가 오른손으로 왼손을 공격했습니다."
    elif ai_action == 3:
        state_temp = atk(state_temp, 1, 1)
        action_str = f"AI가 오른손으로 오른손을 공격했습니다."
    elif ai_action <= 7:
        state_temp = move(state_temp, -(ai_action - 3))
        action_str = f"AI가 왼손으로 -{ai_action - 3}을 이동했습니다."
    else:
        state_temp = move(state_temp, ai_action - 7)
        action_str = f"AI가 왼손으로 {ai_action - 7}을 이동했습니다."
    tk.messagebox.showinfo(title='알림', message=action_str)
    state_temp = state_temp[::-1]
    return state_temp

# 승패 확인
def check(state):
    if (state[0, :] == [0, 0]).all():
        tk.messagebox.showinfo(title='패배!', message='패배했습니다! 진행 상황을 초기화합니다.')
        state = np.array([[1, 1], [1, 1]])
    elif (state[1, :] == [0, 0]).all():
        tk.messagebox.showinfo(title='승리!', message='승리했습니다! 진행 상황을 초기화합니다.')
        state = np.array([[1, 1], [1, 1]])
    return state

# GUI
class Atk_gui(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.selected_p1 = tk.StringVar()
        self.selected_p2 = tk.StringVar()
        self.geometry('300x250')
        self.title('공격')
        ttk.Label(self, text="사용할 손과 공격할 손을 선택하세요").pack(expand=True)

        ttk.Label(self, text="내 손:").pack(expand=True)
        self.selected_p1 = tk.StringVar()
        r1 = ttk.Radiobutton(self, text='왼손(현재 {}개)'.format(self.parent.state[0,0]), value=0, variable=self.selected_p1)
        r1.pack(fill='x', padx=5, pady=5)
        r2 = ttk.Radiobutton(self, text='오른손(현재 {}개)'.format(self.parent.state[0,1]), value=1, variable=self.selected_p1)
        r2.pack(fill='x', padx=5, pady=5)

        ttk.Label(self, text="상대 손:").pack(expand=True)
        self.selected_p2 = tk.StringVar()
        r1 = ttk.Radiobutton(self, text='왼손(현재 {}개)'.format(self.parent.state[1,0]), value=0, variable=self.selected_p2)
        r1.pack(fill='x', padx=5, pady=5)
        r2 = ttk.Radiobutton(self, text='오른손(현재 {}개)'.format(self.parent.state[1,1]), value=1, variable=self.selected_p2)
        r2.pack(fill='x', padx=5, pady=5)

        ttk.Button(self,
                text='확인',
                command=self.sendresult).pack(expand=True)
        ttk.Button(self,
                text='취소',
                command=self.destroy).pack(expand=True)

    def sendresult(self):
        sel_p1 = int(self.selected_p1.get())
        sel_p2 = int(self.selected_p2.get())
        if self.parent.state[0, sel_p1] == 0:
            tk.messagebox.showerror(title='오류', message='0인 손으로 공격할 수 없습니다.')
        elif self.parent.state[1, sel_p2] == 0:
            tk.messagebox.showerror(title='오류', message='0인 손을 공격할 수 없습니다.')
        else:
            self.parent.state = atk(self.parent.state, sel_p1, sel_p2)
            self.parent.update_state()
            self.parent.state = check(self.parent.state)
            self.parent.update_state()
            if (self.parent.state != np.array([[1, 1], [1, 1]])).any():
                self.after(0, self.parent.ai_turn(self.parent.state))
                self.parent.state = check(self.parent.state)
                if (self.parent.state == np.array([[1, 1], [1, 1]])).all():
                    self.parent.turn_choice()
            else:
                self.parent.turn_choice()
            self.parent.update_state()
            self.destroy()

class Move_gui(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.geometry('300x200')
        self.title('이동')
        ttk.Label(self, text="왼손에 옮길 양을 선택하세요").pack(expand=True)

        self.value = tk.StringVar()
        spin_box = ttk.Spinbox(
            self,
            from_=-min(self.parent.state[0, 0], 4 - self.parent.state[0, 1]),  # 오른손에서 왼손으로 옮길 수 있는 최대 양을 계산
            to=min(self.parent.state[0, 1], 4 - self.parent.state[0, 0]),  # 왼손에서 오른손으로 옮길 수 있는 최대 양을 계산
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
        value = int(self.value.get())
        if value == 0 or value == '':
            tk.messagebox.showerror(title='오류', message='0을 선택할 수 없습니다.')
        elif self.parent.state[0, 0] == move(self.parent.state, value)[0, 1]:
            tk.messagebox.showerror(title='오류', message='갯수의 위치만 바뀌게 이동할 수 없습니다.')
        else:
            self.parent.state = move(self.parent.state, value)
            self.parent.state = check(self.parent.state)
            self.parent.update_state()
            if (self.parent.state != np.array([[1, 1], [1, 1]])).any():
                self.parent.state = check(self.parent.state)
                if (self.parent.state == np.array([[1, 1], [1, 1]])).all():
                    self.parent.turn_choice()
            else:
                self.parent.turn_choice()
            self.parent.update_state()
            self.destroy()
            self.after(0, self.parent.ai_turn(self.parent.state))

class Game(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("젓가락 게임")
        self.geometry("300x200")
        self.state = np.array([[1, 1], [1, 1]])
        self.turn = 0

        self.state_label = ttk.Label(self)
        self.state_label.pack(fill='x', padx=5, pady=5)
        self.state_label.config(text=f"내 상태: 왼손 {self.state[0,0]}, 오른손 {self.state[0,1]}\n상대 상태: 왼손 {self.state[1,0]}, 오른손 {self.state[1,1]}")

        ttk.Button(self,
                text='공격',
                command=self.atk).pack(fill='x', padx=5, pady=5)
        ttk.Button(self,
                text='이동',
                command=self.move).pack(fill='x', padx=5, pady=5)
        self.turn_choice()

    def turn_choice(self):
        choice = tk.messagebox.askyesno("선공 선택", "선공을 하시겠습니까?")
        if choice:
            self.turn = 0
        else:
            self.turn = 1
            self.ai_turn(self.state)

    def update_state(self):
        self.state_label.config(text=f"내 상태: 왼손 {self.state[0,0]}, 오른손 {self.state[0,1]}\n상대 상태: 왼손 {self.state[1,0]}, 오른손 {self.state[1,1]}")

    def atk(self):
        Atk_gui(self)
        self.update_state()

    def move(self):
        Move_gui(self)
        self.update_state()

    def ai_turn(self, state):
        self.state = ai_action(state)
        self.update_state()
        
if __name__ == "__main__":
    app = Game()
    app.mainloop()