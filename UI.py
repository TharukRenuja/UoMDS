import tkinter as tk
from tkinter import ttk
import Chatbot
import DeepSearch

window = tk.Tk()
window.title('AI Chatbot')
window.geometry('500x650')

label_1 = ttk.Label(text='Your Questions: ')
label_1.pack(pady=5)


inputField = tk.Text(height=10)
inputField.pack(padx=50,pady=10)

outputField = tk.Text(height=10)

def processAnswers():
    outputField.delete('1.0', tk.END)
    outputField2.delete('1.0', tk.END)
    query = inputField.get('1.0', tk.END).split('\n')
    answer = ''
    for question in query:
        question = question.strip()
        if not question:
            continue

        print(f"Processing: {question}")
        chunkGen = Chatbot.retrieve_relevant_chunks(question, Chatbot.collection, Chatbot.embedding_model_name)
        if chunkGen:
            ai_answer = Chatbot.generate_ai_answer(question, chunkGen, Chatbot.generativeModel)
            deep_search_Answer = DeepSearch.main(''.join(query))
            outputField.insert(tk.END, ai_answer + "\n\n\n")
            outputField2.insert(tk.END, f"Research Response:\nTitle: {deep_search_Answer.title}\nSummary: {deep_search_Answer.summary}\nSources: {deep_search_Answer.sources}\nTools used: {deep_search_Answer.tools}" + "\n\n\n")
        else:
            outputField.insert(tk.END, "Error: No relevant chunks found.\n\n")


processButton = ttk.Button(text='Answer Me!',command=processAnswers)
processButton.pack()

label_2 = ttk.Label(text='Your Answers: ')
label_2.pack(pady=5)

outputField.pack(padx=50,pady=10)

label_3 = ttk.Label(text='Deep search: ')
label_3.pack(pady=5)

outputField2 = tk.Text(height=10)
outputField2.pack(padx=50,pady=10)

tk.mainloop()