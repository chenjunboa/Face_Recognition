import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk
import sys

def lojin():
    success = []

    # 主窗口
    window = tk.Tk()
    window.title("Please Log In")
    window.geometry("900x660")
    sw = window.winfo_screenwidth()
    # 得到屏幕宽度
    sh = window.winfo_screenheight()
    x = (sw - 900) / 2
    y = (sh - 660) / 2
    window.geometry("%dx%d+%d+%d" % (900, 660, x, y))
    window.resizable(0, 0)  # 禁止缩放
    window.overrideredirect(True)
    # 背景画布
    canvas = tk.Canvas(window, width=1000, height=500)
    image_file = ImageTk.PhotoImage(file="beijin.jpg")
    image = canvas.create_image(0, 0, anchor="nw", image=image_file)
    canvas.pack()

    # 用户名，用户密码文本
    name_lable = tk.Label(window, text="用户名:", font=(15))
    password_lable = tk.Label(window, text="密码:", font=(15))
    name_lable.place(x=300, y=510)
    password_lable.place(x=300, y=550)

    # 用户名，用户密码输入框
    nameval = tk.StringVar()
    passwordval = tk.StringVar()
    name_entry = tk.Entry(window, textvariable=nameval, font=(12))
    password_entry = tk.Entry(window, textvariable=passwordval, show="*", font=(12))
    name_entry.place(x=400, y=515)
    password_entry.place(x=400, y=555)

    # 登录按钮触发函数
    def sign_in_f():

        user_si_name = name_entry.get()
        user_si_pass = password_entry.get()
        if user_si_name in success:
            i = success.index(user_si_name)
            if success[i + 1] == user_si_pass:
                tk.messagebox.showinfo(title="登录提示", message="登录成功")
                window.destroy()
                return 1
                pass

            else:
                tk.messagebox.showinfo(title="登录提示", message="密码错误")
        else:
            result = tk.messagebox.askquestion(title="登录提示", message="用户名不存在,是否立即注册？")
            if result == "yes":
                sign_up_f()
            else:
                pass

    # 注册按钮触发函数
    def sign_up_f():
        # 用户注册界面
        window.withdraw()
        singn_up_w = tk.Tk()
        singn_up_w.title("用户注册")
        singn_up_w.geometry("600x400")
        # 禁止缩放
        w = singn_up_w.winfo_screenwidth()
        # 得到屏幕宽度
        h = singn_up_w.winfo_screenheight()
        x = (w - 600) / 2
        y = (h - 400) / 2
        singn_up_w.geometry("%dx%d+%d+%d" % (600, 400, x, y))
        singn_up_w.resizable(0, 0)  # 禁止缩放
        singn_up_w.overrideredirect(True)
        # 拥护注册 用户名，密码，确认密码文本
        su_name_lable = tk.Label(singn_up_w, text="用户名:", font=(12))
        su_pass_lable = tk.Label(singn_up_w, text="密码:", font=(12))
        su_cpass_lable = tk.Label(singn_up_w, text="确认密码:", font=(12))
        su_name_lable.place(x=95, y=50)
        su_pass_lable.place(x=95, y=150)
        su_cpass_lable.place(x=95, y=250)

        # 用户注册 用户名，密码，确认密码输入框
        su_name_val = tk.StringVar()
        su_pass_val = tk.StringVar()
        su_cpass_val = tk.StringVar()
        su_name_entry = tk.Entry(singn_up_w, textvariable=su_name_val, width=20, font=(12))
        su_pass_entry = tk.Entry(singn_up_w, textvariable=su_pass_val, width=20, show="*", font=(12))
        su_cpass_entry = tk.Entry(singn_up_w, textvariable=su_cpass_val, width=20, show="*", font=(12))
        su_name_entry.place(x=270, y=50)
        su_pass_entry.place(x=270, y=150)
        su_cpass_entry.place(x=270, y=250)

        # 用户在注册页面点击注册按钮触发的函数
        def su_conf_b():
            su_username = su_name_entry.get()
            su_userpass = su_pass_entry.get()
            su_usercpass = su_cpass_entry.get()
            if su_userpass == su_usercpass:
                tk.messagebox.showinfo(title="注册提示", message="注册成功，请登录")
                window.deiconify()
                success.append(su_username)
                success.append(su_userpass)
                singn_up_w.destroy()

            else:
                tk.messagebox.showinfo(title="注册提示", message="两次输入的密码不同，请重新输入")

        # 用户在注册页面点击取消按钮触发的函数
        def su_cancel_b():
            result = tk.messagebox.askquestion(title="放弃注册", message="你真的要放弃注册吗？")
            if result == "yes":
                singn_up_w.destroy()
            else:
                pass

        # 用户注册 注册，取消按钮
        su_confirm_button = tk.Button(singn_up_w, text="Sign up", command=su_conf_b)
        su_cancle_button = tk.Button(singn_up_w, text="Cancel", command=su_cancel_b)
        su_confirm_button.place(x=170, y=330)
        su_cancle_button.place(x=370, y=330)

    # 登录，注册按钮
    sign_in_button = tk.Button(window, text="Sign in", command=sign_in_f)
    sign_up_button = tk.Button(window, text="Sign up", command=sign_up_f)
    sign_in_button.place(x=350, y=600)
    sign_up_button.place(x=470, y=600)

    window.mainloop()
