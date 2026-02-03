import os

def create_dummy_data():
    base_path = os.path.join("data", "train")
    
    # Author A: Formal, Academic, Complex sentences
    author_a_texts = [
        "The automated analysis of linguistic patterns presents a significant advancement in forensic methodology.",
        "Methodological rigor is essential when evaluating the statistical significance of stylometric features.",
        "This study aims to elucidate the underlying structures governing authorial distinctiveness in digital texts.",
        "Furthermore, the correlation between lexical density and cognitive complexity cannot be overstated.",
        "In conclusion, the results demonstrate a robust adherence to standard grammatical conventions.",
        "The proliferation of digital communication necessitates novel approaches to identity verification.",
        "Abstract theoretical frameworks often precede practical implementation in computational linguistics.",
        "We observe a distinct deviation from the norm in the subject's usage of subordinate clauses.",
        "The hypothesis was tested using a multivariate analysis of variance on the extracted feature set.",
        "Consequently, the data suggests a high probability of single-authorship across the provided samples."
    ]
    
    # Author B: Informal, Chatty, Simple sentences, slang
    author_b_texts = [
        "Hey, did you see that new movie? It was totally awesome and super cool!",
        "I'm gonna go to the store later, wanna come with me? It'll be fun.",
        "lol that's so funny, I can't believe he actually said that to you.",
        "Just woke up, feeling super tired today. Need coffee ASAP!!!",
        "omg, you won't believe what happened at the party last night.",
        "So yeah, I was thinking we should maybe grab some pizza or something.",
        "I dunno, it just seems kinda weird to me, you know what I mean?",
        "Text me later when you get home, k? Don't forget!",
        "Wow, this weather is crazy. One minute it's sunny, then it's raining.",
        "Haha for real? That's insane. calling you in 5 mins."
    ]

    os.makedirs(os.path.join(base_path, "Author_A"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "Author_B"), exist_ok=True)

    for i, text in enumerate(author_a_texts):
        with open(os.path.join(base_path, "Author_A", f"doc_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
            
    for i, text in enumerate(author_b_texts):
        with open(os.path.join(base_path, "Author_B", f"doc_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    print("Dummy data generated successfully.")

if __name__ == "__main__":
    create_dummy_data()
