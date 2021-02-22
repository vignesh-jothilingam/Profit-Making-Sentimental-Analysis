import pickle
import tkinter as tk
from tkinter import ttk, Canvas, NW
import tkinter.font as tkFont




window = tk.Tk()
style = ttk.Style()
style.configure("BW.TLabel",foreground="white",background="black")


window.title("Profit Making Sentimental Analysis")
window.minsize(800, 600)
window.configure(bg='blue')


titleStyle = tkFont.Font(family="Lucida Grande", size=23)
resultStyle = tkFont.Font(family="Lucida Grande", size=21)
fontStyle = tkFont.Font(family="Lucida Grande", size=24)



def predict_report():
    text = input_val.get()
    classifier = pickle.load(open('classifier.sav', 'rb'))
    vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
    text_vector = vectorizer.transform([text])
    result = classifier.predict(text_vector)

    
    ans = result[0]
    
    window.update_idletasks()
    if str(ans)=='neg':
        resultTx.configure(text="Sentimental Analysis : Negative")
        result_page_Txt.configure(text="Customer is unsatisfied this prodcut")
        result_page_Txt2.configure(text="It will not give profit, but gives lose")
    elif str(ans)=='pos':
        resultTx.configure(text="Sentimental Analysis : Positive")
        result_page_Txt.configure(text="Customer is satisfied this prodcut")
        result_page_Txt2.configure(text="It will give profit in future")
    
label = ttk.Label(window, text="Profit Making Sentimental Analysis", font=titleStyle)
label.grid(column=0, row=0)
label.place(x=400, y=52, anchor="center")



input_val = tk.StringVar(value='')
nameEntered = ttk.Entry(window, width=35, textvariable=input_val, font=('Verdana',20))
nameEntered.grid(column=1, row=1)
nameEntered.place(x=400, y=192, anchor="center")

button = ttk.Button(window, text="Predict and report", command=predict_report)
button.grid(column=0, row=2)
button.place(x=400, y=280, anchor="center",width=160,height=30)

resultTx = ttk.Label(window, text="", font=resultStyle)
resultTx.grid(column=0, row=0)
resultTx.place(x=400, y=380, anchor="center")

result_page_Txt = ttk.Label(window, text="", font=resultStyle)
result_page_Txt.grid(column=0, row=0)
result_page_Txt.place(x=400, y=460, anchor="center")

result_page_Txt2 = ttk.Label(window, text="", font=resultStyle)
result_page_Txt2.grid(column=0, row=0)
result_page_Txt2.place(x=400, y=560, anchor="center")


window.mainloop()
