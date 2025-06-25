from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain

import langchain

from lang_chain_project.utils import data_loader, open_ai

import warnings
warnings.filterwarnings("ignore")

loader = data_loader.load_csv_data_as_doc_loader()
data = loader.load()
# Coming up with test datapoints
print(f"data[10]:\n{data[10]}")
print(f"data[11]:\n{data[11]}")
print(f"=================================================================================================")

llm_model = "deepseek-r1:1.5b"
embedding = open_ai.NewEmbeddings(model=llm_model)
llm = open_ai.NewLangChainChatOpenAI(model=llm_model)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding,
).from_loaders([loader])

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

# Hard-coded examples
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# LLM-Generated examples
example_gen_chain = QAGenerateChain.from_llm(llm=llm)
# the warning below can be safely ignored
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

print(f"new_examples[0]: \n{new_examples[0]}")
print(f"data[0]:\n{data[0]}")
print(f"=================================================================================================")

# Combine the hard-coded and LLM-generated examples
examples += new_examples
qa.run(examples[0]["query"])

# Manual Evaluation
langchain.debug = True
qa.run(examples[0]["query"])
langchain.debug = False

# LLM assisted evaluation
predictions = qa.apply(examples)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)

for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()

print(f"graded_outputs[0]:\n{graded_outputs[0]}")