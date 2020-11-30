import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()

doc = nlp("I live in New York")
print("Before:", [token.text for token in doc])

with doc.retokenize() as retokenizer:
    retokenizer.merge(doc[3:5], attrs={"LEMMA": "new york"})
print("After:", [token.text for token in doc])

for token in nlp("Let's go to NY!"):
    print(token.text)

for ent in nlp("Apple is looking at buying U.K. startup for $1 billion").ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

for chunk in nlp("Autonomous cars shift insurance liability toward manufacturers").noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text)

for token in nlp("Apple is looking at buying U.K. startup for $1 billion"):
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop)

# document level
doc = nlp("Ada Lovelace was born in London")
ents = [(e.text, e.label_, e.kb_id_) for e in doc.ents]
print(ents)  # [('Ada Lovelace', 'PERSON', 'Q7259'), ('London', 'GPE', 'Q84')]

# token level
ent_ada_0 = [doc[0].text, doc[0].ent_type_, doc[0].ent_kb_id_]
ent_ada_1 = [doc[1].text, doc[1].ent_type_, doc[1].ent_kb_id_]
ent_london_5 = [doc[5].text, doc[5].ent_type_, doc[5].ent_kb_id_]
print(ent_ada_0)  # ['Ada', 'PERSON', 'Q7259']
print(ent_ada_1)  # ['Lovelace', 'PERSON', 'Q7259']
print(ent_london_5)  # ['London', 'GPE', 'Q84']

