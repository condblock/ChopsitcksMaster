# 함수 선언
def check(): # check win or lose
    global p1
    global p2
    if p1['l'] == 0 and p1['r'] == 0:
        p1 = {'l': 1, 'r': 1}
        p2 = {'l': 1, 'r': 1}
        return "win"
    elif p2['l'] == 0 and p2['r'] == 0:
        p1 = {'l': 1, 'r': 1}
        p2 = {'l': 1, 'r': 1}
        return "lose"
        

# atk, move function
# key는 l 또는 r
def atk(atkdict, atk_key, victdict, vict_key):
    if atkdict[atk_key] == 0 or victdict[vict_key] == 0:
        return False
    else:
        victdict[vict_key] += atkdict[atk_key]
        if victdict[vict_key] > 5:
            victdict[vict_key] = 0
        return True

def move(dict, key, amount): # dict[key] > 0, result of dict[key] > 0, result of dict[other] <= 5
    if key == 'l':
        key2 = 'r'
    elif key == 'r':
        key2 = 'l'
    if dict[key] + amount > 5:
        return "error1"
    elif dict[key] - amount < 0:
        return "error2"
    elif (dict[key], dict[key2]) == (dict[key2] - amount, dict[key] + amount):
        return "error3"
    else:
        if key == 'l':
            dict['l'] += amount
            dict['r'] -= amount
        elif key == 'r':
            dict['r'] += amount
            dict['l'] -= amount
        return True

# 초기값 설정
p1 = {'l': 1, 'r': 1}
p2 = {'l': 1, 'r': 1}

