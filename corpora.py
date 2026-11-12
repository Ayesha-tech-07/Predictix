"""Built-in training texts for PredictiX.

All three passages are original, written for this project, so there are no
copyright concerns. They are deliberately repetitive in places (the same
phrases come back in different contexts) because N-gram models learn from
repetition: the more often a word sequence appears, the better the model
can predict it.
"""

CAMPUS = """
The students walk to the library every morning. The library opens early and closes late.
Every student carries a laptop and a notebook. The teacher explains the lesson to the students.
The students ask questions and the teacher answers them patiently. In the morning the students attend the lecture.
In the afternoon the students work on their projects in the lab. The lab is quiet and the computers are fast.
The students build small projects and test them carefully. Every project needs a clear idea and a good plan.
The teacher reviews the project and gives useful feedback. The students improve the project after the feedback.
Natural language processing is an interesting subject. The students learn natural language processing in the third year.
Natural language processing helps computers understand human language. Machine learning helps computers learn from data.
The students practice machine learning with small datasets. The exam is difficult but the students prepare well.
The students revise the notes before the exam. Good notes make the revision easy.
The teacher shares the notes with the students. The students share the notes with their friends.
Friends help each other during the exam week. The canteen is busy at noon.
The students eat lunch and talk about the project. After lunch the students go back to the lab.
The evening is calm and the campus is green. Some students read books under the trees.
Some students practice coding on their laptops. The coding club meets every friday.
The club members build apps and share ideas. A good app solves a real problem.
The students present their apps to the teacher. The teacher likes the apps and praises the effort.
The students feel proud of their work. Hard work brings good results.
Every student wants to learn and grow. The college organizes a technical fest every year.
The fest has coding contests and project exhibitions. The winners receive prizes and certificates.
The students enjoy the fest and make new friends.
"""

TECHNOLOGY = """
Artificial intelligence is changing the world. Machine learning models learn patterns from large datasets.
A language model predicts the next word in a sentence. A good language model understands the context of the sentence.
Natural language processing is used in chatbots and virtual assistants. Virtual assistants answer questions and set reminders.
Search engines use language models to understand the query. The query is matched with the most relevant documents.
Translation systems convert text from one language to another. Speech recognition systems convert speech into text.
Text messages are completed by predictive keyboards. The keyboard suggests the next word while the user types.
Large datasets make the model more accurate. Clean data makes the model more reliable.
The model is trained on the data and tested on new sentences. The model is evaluated using perplexity.
A lower perplexity means a better model. Developers build applications that use language models.
These applications help people write faster. Cloud platforms make it easy to deploy applications.
A web application can be shared with anyone through a link. Python is a popular language for machine learning.
Python has many libraries for data analysis. Developers use python to build apps and train models.
Data scientists clean the data and visualize the results. A clear chart explains the result better than a long table.
Privacy is important when models learn from personal data. Responsible developers protect the privacy of the users.
The future of artificial intelligence depends on good data and good ideas. New models become faster and smarter every year.
Many companies use artificial intelligence to improve their services. Hospitals use artificial intelligence to support doctors.
Schools use artificial intelligence to help students learn. Farmers use artificial intelligence to predict the weather.
Artificial intelligence will help people solve difficult problems.
"""

TRAVEL = """
We travel to the mountains every summer. The mountains are cold and the air is fresh.
The train ride is long but the view is beautiful. We carry warm clothes and a small camera.
The village is quiet and the people are kind. The village market sells fresh fruit and hot tea.
We drink hot tea and watch the sunrise. The sunrise paints the sky in pink and gold.
In the evening we walk along the river. The river is clear and the water is cold.
We cook simple food on a small stove. The food tastes better in the open air.
Hot rice and spicy curry make a perfect dinner. My favorite dessert is sweet rice with milk.
The city has many good restaurants. The restaurants serve biryani and fresh bread.
The biryani is spicy and full of flavor. We eat biryani with cool yogurt.
Street food is popular in the city. Street food vendors sell tea and fried snacks.
The snacks are crispy and hot. On weekends we visit the old fort.
The old fort stands on a high hill. The hill gives a wide view of the city.
We take photos and share them with friends. Travel teaches us about new places and new people.
Every journey creates a story to remember. A good journey needs a clear plan and a happy mind.
We plan the trip and pack our bags. The bus leaves early in the morning.
We reach the beach by noon. The beach is warm and the waves are gentle.
Children build sand castles near the water. We swim in the sea and rest under the trees.
The sunset over the sea is a beautiful sight. We return home tired but happy.
"""

CORPORA = {
    "Campus life": CAMPUS,
    "Technology and AI": TECHNOLOGY,
    "Travel and food": TRAVEL,
}

DEFAULT_CORPUS = "Campus life"
