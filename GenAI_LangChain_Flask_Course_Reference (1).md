# GenAI, LangChain & Flask — Complete Course Reference

*A consolidated study reference covering Generative AI foundations, NLP, Prompt Engineering, LangChain, and Flask web development.*

---

## Table of Contents

1. [Foundations of Generative AI](#1-foundations-of-generative-ai)
2. [Natural Language Processing (NLP) Basics](#2-natural-language-processing-nlp-basics)
3. [Prompt Engineering](#3-prompt-engineering)
4. [LangChain Core Concepts](#4-langchain-core-concepts)
5. [LangChain Expression Language (LCEL)](#5-langchain-expression-language-lcel)
6. [LangChain Chains, Memory & Agents](#6-langchain-chains-memory--agents)
7. [Document Loaders, Splitters, Embeddings & Retrieval (RAG)](#7-document-loaders-splitters-embeddings--retrieval-rag)
8. [Flask Web Framework](#8-flask-web-framework)
9. [Building a GenAI Flask Application (IBM watsonx + LangChain)](#9-building-a-genai-flask-application-ibm-watsonx--langchain)
10. [The GenAI Application Development Lifecycle](#10-the-genai-application-development-lifecycle)
11. [Choosing the Right Model for the Right Use Case](#11-choosing-the-right-model-for-the-right-use-case)
12. [Quick-Reference Cheat Sheets](#12-quick-reference-cheat-sheets)
13. [Retrieval-Augmented Generation (RAG) Fundamentals](#13-retrieval-augmented-generation-rag-fundamentals)
14. [LlamaIndex](#14-llamaindex)
15. [LangChain vs. LlamaIndex — Detailed Comparison](#15-langchain-vs-llamaindex--detailed-comparison)
16. [Gradio — Building Web UIs for Models](#16-gradio--building-web-uis-for-models)
17. [Course 2 Cheat Sheets & Appendix](#17-course-2-cheat-sheets--appendix)

---

## 1. Foundations of Generative AI

### 1.1 Discriminative AI vs. Generative AI

- **AI** = simulation of human intelligence by machines. Models learn from data through a process called **training**.
- **Discriminative AI**: learns to distinguish between classes of labeled data (a decision boundary). Best for classification tasks (e.g., spam filters). Cannot generate new content or understand context beyond its labels.
- **Generative AI**: learns the underlying distribution of training data and generates **new, original content** (text, images, audio, video, code, data). It can transform input types (text→image, text→text, image→video, etc.).
  - Example: Discriminative AI answers "Is this a drawing of a nest or an egg?" Generative AI responds to "Draw a nest with three eggs in it."
- Both discriminative and generative models are built using **deep learning** (artificial neural networks — collections of "neurons" modeled loosely on the brain).

### 1.2 Generative Model Families (the "building blocks")

- **GANs** (Generative Adversarial Networks) — introduced 2014 by Ian Goodfellow.
- **VAEs** (Variational Autoencoders)
- **Transformers**
- **Diffusion Models**

### 1.3 Timeline

| Era | Milestone |
|---|---|
| Late 1950s | Machine learning proposed; algorithms explored for generating new data |
| 1990s | Rise of neural networks advances generative AI |
| Early 2010s | Deep learning + big data + compute accelerates progress |
| 2014 | GANs introduced |
| 2018 | OpenAI introduces GPT (Generative Pre-trained Transformer) |
| Later | GPT-3, GPT-4, Google PaLM, Meta LLaMA, Stable Diffusion, DALL-E |

### 1.4 Foundation Models & LLMs

- **Foundation models** (term coined by a Stanford team): broad, general-purpose models trained once that can be adapted/transferred to many downstream tasks — replacing the old paradigm of one narrow model per task.
- Trained on **huge amounts of unstructured, unlabeled data** in an unsupervised, self-predictive way (e.g., predicting the next word: "no use crying over spilled ___ → milk"). This next-token prediction is why foundation models fall under **Generative AI**.
- **Large Language Models (LLMs)** are a category of foundation model specialized for human language (understanding + generating text).
- Adapting a foundation model to a specific task can be done via:
  - **Tuning**: introduce a small amount of labeled data and update model parameters/weights.
  - **Prompting / Prompt Engineering**: use the model as-is, guiding it with input text — works well even with little or no labeled data.

**Advantages of foundation models**
- **Performance**: massive pretraining data lets them outperform models trained from scratch on small task-specific datasets.
- **Productivity**: far less labeled data needed to reach a working task-specific model (via prompting/tuning) versus training from zero.

**Disadvantages**
- **Compute cost**: expensive to train and to run inference (often requiring multiple GPUs), which is a barrier for smaller organizations.
- **Trustworthiness**: trained on huge amounts of scraped, often unvetted internet data — risk of bias, toxicity, and unclear provenance ("we often don't even know what data some open-source models were trained on").

**Applications beyond language** (IBM examples): vision (DALL·E 2), code (Copilot), chemistry (Moleformer for molecule discovery), climate (earth-science foundation models using geospatial data), enterprise products (Watson Assistant, Watson Discovery, Maximo Visual Inspection, Project Wisdom with Red Hat/Ansible).

---

## 2. Natural Language Processing (NLP) Basics

- **NLP** = teaching a computer to process human language. It sits between:
  - **Unstructured text** (how humans naturally speak/write) and
  - **Structured data** (a format a computer can act on, e.g., JSON-like objects).
- **NLU (Natural Language Understanding)**: unstructured → structured.
- **NLG (Natural Language Generation)**: structured → unstructured.

### 2.1 Common NLP Use Cases

- **Machine translation** — must preserve context, not just word-for-word translation (classic failure example: "the spirit is willing but the flesh is weak" round-tripped through Russian becomes something like "the vodka is good but the meat is rotten").
- **Virtual assistants & chatbots** — convert utterances/text into an actionable command or decision-tree traversal.
- **Sentiment analysis** — determine positive/negative/sarcastic tone in reviews, emails, etc.
- **Spam detection** — flags based on signals like overused words, poor grammar, false urgency.

### 2.2 The NLP "Bag of Tools" Pipeline

1. **Tokenization** — break a string into chunks/tokens (e.g., "add eggs and milk to my shopping list" → 8 tokens).
2. **Stemming** — reduce a token to its crude root by stripping prefixes/suffixes (running, runs, ran → run). Doesn't always work well (university ≠ universe).
3. **Lemmatization** — find a token's true dictionary root/lemma using its meaning (better → good, since "better" derives from "good"; the raw *stem* of "better" would incorrectly be "bet").
4. **Part-of-Speech (POS) tagging** — determine a word's grammatical role from context (e.g., "make" is a verb in "I'm going to make dinner" but a noun in "what make is your laptop?").
5. **Named Entity Recognition (NER)** — detect entities associated with a token (e.g., "Arizona" → U.S. state; "Ralph" → person's name).

---

## 3. Prompt Engineering

### 3.1 What Is a Prompt?

A **prompt** is the instruction/input given to an LLM to guide it toward a task or output. Two core components:
- **Instructions** — clear, direct commands telling the AI what to do.
- **Context** — background information/data/parameters that shape the response.

A fully structured prompt has **four elements**:
1. **Instructions** — what to do (e.g., "classify this review as positive, negative, or neutral").
2. **Context** — background (e.g., "this review is feedback for a newly launched product").
3. **Input data** — the actual content to process (e.g., the review text itself).
4. **Output indicator** — marks where/how the answer should appear (e.g., "Sentiment:").

### 3.2 Prompt Engineering — Why It Matters

- Boosts **effectiveness and accuracy** of LLM outputs.
- Ensures **relevance** to context.
- Helps **meet user expectations** with fewer misunderstandings.
- Reduces/eliminates the need for **continual fine-tuning**.

### 3.3 In-Context Learning

- A method of prompt engineering where task **demonstrations are given directly in the prompt** (natural language), with **no additional training/fine-tuning** required.
- **Pros**: no fine-tuning needed → saves time/resources.
- **Cons**: limited by how much can realistically fit in the context window; very complex tasks may still need real training (gradient-based weight updates).

### 3.4 Prompting Techniques

| Technique | Description |
|---|---|
| **Zero-shot prompting** | Ask the model to perform a task with **no examples** at all. Relies purely on the model's pretrained knowledge. Example: "Classify: 'The Eiffel Tower is located in Berlin.' True or False?" |
| **One-shot prompting** | Give the model **exactly one example** as a template before asking it to do a similar new task (e.g., show one English→French translation, then ask for a new one). |
| **Few-shot prompting** | Give **multiple examples (typically 2–5)** to establish a clearer pattern before the real task (e.g., 3 labeled emotion examples, then classify a new statement). |
| **Chain-of-Thought (CoT) prompting** | Ask the model to reason **step-by-step** through a problem before answering. Highly effective for multi-step arithmetic/logic. Example: apples bought/sold/delivered word problem, solved via visible intermediate steps. |
| **Self-consistency** | Generate **multiple independent reasoning paths/answers** to the same question, then compare them to settle on the most consistent, reliable result. Example: age-riddle solved three different ways to cross-check the answer. |

### 3.5 Tools for Prompt Engineering

OpenAI's Playground, LangChain, Hugging Face Model Hub, IBM's AI Classroom — these let you develop, test, tweak, share, and analyze prompts/results in real time, and access pretrained models for many tasks/languages.

### 3.6 Prompt Engineering in LangChain

LangChain provides **prompt templates** — reusable "recipes" that bundle:
- Instructions for the model,
- Optional few-shot examples,
- A specific question/placeholder for input.

```python
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a {adjective} joke about {content}")
formatted_prompt = prompt_template.format(adjective="funny", content="chickens")
# -> "Tell me a funny joke about chickens"
```

**Example selectors** (used inside few-shot prompt templates to pick the most relevant examples from a larger example library):
- Semantic Similarity
- Max Marginal Relevance (for diversity)
- "Examples of Efficient Prompts"
- N-Gram Overlap (textual similarity)

**Agents** (see Section 6) are the mechanism by which prompt-engineered LLMs are combined with tools to perform complex, multi-domain tasks: Q&A-with-sources agents, content-creation/summarization agents, analytics agents (BI), multilingual/translation agents.

---

## 4. LangChain Core Concepts

### 4.1 What Is LangChain?

An **open-source framework/interface** (Python) that simplifies building LLM-powered applications. It provides components and interfaces to integrate LLMs into apps — for NLP, data retrieval, summarization, Q&A, and more. It "chains" together retrieval, extraction, processing, and generation steps — hence the name.

**Key benefits**
- **Modularity** — components snap together like building blocks; encourages reuse.
- **Extensibility** — add new features / integrate external systems with minimal codebase changes.
- **Decomposition** — breaks complex tasks into smaller manageable steps (mimicking human problem solving), improving inference accuracy.
- **Vector database integration** — enables fast semantic search over large datasets.

**Practical uses**: content summarization (e.g., legal documents), data/statistic extraction from reports, sophisticated Q&A systems (customer support), automated content generation (emails, brainstorming, technical docs). It can also work with non-text data (images, audio, video) via external libraries/models (e.g., speech-to-text) combined with vector embeddings.

### 4.2 The Core Components

LangChain is built from: **Documents, Chains, Agents, Language Model, Chat Model, Chat Message, Prompt Templates, Output Parsers.**

#### Language Model
- Foundation for LLMs: takes text in, produces text out (completion, summarization, etc.).
- LangChain supports IBM, OpenAI, Google, and Meta models as primary providers.

```python
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

model_id = 'mistralai/mixtral-8x7b-instruct-v01'
parameters = {GenParams.MAX_NEW_TOKENS: 256, GenParams.TEMPERATURE: 0.2}
credentials = {"url": "https://us-south.ml.cloud.ibm.com"}
project_id = "skills-network"

model = ModelInference(model_id=model_id, params=parameters,
                        credentials=credentials, project_id=project_id)
mixtral_llm = WatsonxLLM(model=model)
response = mixtral_llm.invoke("Who is man's best friend?")
```

#### Chat Model
- A language model specialized for **conversation**: understands prompts/questions and responds like a human, using structured message types.

#### Chat Messages
Chat models process several message types:

| Message type | Purpose |
|---|---|
| **HumanMessage** | user input |
| **AIMessage** | model-generated output |
| **SystemMessage** | instructs/configures the model's behavior |
| **FunctionMessage** | represents the outcome of a called function (has a `name` param) |
| **ToolMessage** | represents the result of a tool interaction |

Each chat message has two key properties: **role** (who's speaking) and **content** (what's said).

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

msg = mixtral_llm.invoke([
    SystemMessage(content="You are a helpful AI bot that assists a user in choosing the perfect book to read in one short sentence"),
    HumanMessage(content="I enjoy mystery novels, what should I read?")
])
```
You can also simulate a past conversation (human + AI messages) before asking a new question, or skip system/AI messages entirely for a direct human→model interaction.

#### Prompt Templates
Translate user questions into clear instructions for the LLM. Types:

| Template type | Purpose |
|---|---|
| **String prompt template** (`PromptTemplate`) | single-string formatting |
| **Chat prompt template** (`ChatPromptTemplate`) | formats a list of role-tagged messages |
| **Message prompt templates** | `AIMessagePromptTemplate`, `SystemMessagePromptTemplate`, `HumanMessagePromptTemplate`, `ChatMessagePromptTemplate` — flexible role assignment |
| **MessagesPlaceholder** | full control over inserting a list of messages at a specific spot |
| **Few-Shot Prompt Template** | injects specific examples/shots to guide the LLM |

```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])
formatted_messages = prompt.invoke({"topic": "cats"})
```

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])
formatted_messages = prompt.invoke({"msgs": [HumanMessage(content="What is the day after Tuesday?")]})
```

#### Output Parsers
Transform raw LLM text output into structured formats: **JSON, XML, CSV, pandas DataFrames**, etc.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

output_parser = JsonOutputParser(pydantic_object=Joke)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)
chain = prompt | mixtral_llm | output_parser
```

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Answer the user query. {format_instructions}\nList five {subject}.",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)
chain = prompt | mixtral_llm | output_parser
result = chain.invoke({"subject": "ice cream flavors"})
```

---

## 5. LangChain Expression Language (LCEL)

**LCEL** is the modern, recommended way to build LangChain pipelines using the **pipe operator (`|`)** to connect components — replacing the older `LLMChain`/`SequentialChain` style. Benefits: better composability, clearer data-flow visualization, more flexibility, plus built-in **parallel execution, async support, simplified streaming, and automatic tracing**.

**Typical LCEL pattern**
1. Define a template with `{variables}`.
2. Create a prompt template instance.
3. Build a chain by piping components together.
4. Invoke the chain with input values.

### 5.1 Runnables

**Runnables** are the interface/building blocks connecting LLMs, retrievers, and tools into a pipeline. Two core composition primitives:

- **RunnableSequence** — chains components sequentially (output of one → input of next). LCEL shortcut: just connect them with `|` instead of instantiating `RunnableSequence` explicitly.
- **RunnableParallel** — runs multiple components **concurrently**, each receiving the *same* input.

### 5.2 Automatic Type Coercion

LCEL automatically converts plain Python objects into runnables:
- A **dictionary** → becomes a `RunnableParallel` (runs multiple tasks simultaneously on the same input).
- A **function** → becomes a `RunnableLambda` (transforms input).

```python
# Dictionary → RunnableParallel: summary, translation, and sentiment run concurrently on the same "text" input
chain = prompt | llm  # example combining a prompt template with the LLM
parallel_chain = {
    "summary": summary_prompt | llm,
    "translation": translation_prompt | llm,
    "sentiment": sentiment_prompt | llm,
}
result = parallel_chain.invoke({"text": some_text})
# result -> {"summary": ..., "translation": ..., "sentiment": ...}
```

### 5.3 RunnableLambda Example — a Simple Joke Chain

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_prompt(variables):
    return prompt.format(**variables)

joke_chain = (
    RunnableLambda(format_prompt)   # wraps the function, formats the prompt with adjective/content
    | llm                             # sends the formatted prompt to the LLM
    | StrOutputParser()               # extracts a clean string from the LLM response
)
response = joke_chain.invoke({"adjective": "funny", "content": "chickens"})
```

Flow: `RunnableLambda` takes the input dict → passes it to `format_prompt` → formatted prompt goes to the LLM → LLM's response goes to `StrOutputParser`.

### 5.4 When to Use LCEL vs. LangGraph

- LCEL is best suited for **simpler orchestration** tasks.
- For **more complex workflows** (branching, cycles, stateful multi-agent flows), use **LangGraph**, while still leveraging LCEL *within* individual nodes.

### 5.5 RunnablePassthrough (multi-step chaining with LCEL)

```python
from langchain_core.runnables import RunnablePassthrough

location_chain_lcel = PromptTemplate.from_template(location_template) | mixtral_llm | StrOutputParser()
dish_chain_lcel     = PromptTemplate.from_template(dish_template)     | mixtral_llm | StrOutputParser()
time_chain_lcel     = PromptTemplate.from_template(time_template)     | mixtral_llm | StrOutputParser()

overall_chain_lcel = (
    RunnablePassthrough.assign(meal=lambda x: location_chain_lcel.invoke({"location": x["location"]}))
    | RunnablePassthrough.assign(recipe=lambda x: dish_chain_lcel.invoke({"meal": x["meal"]}))
    | RunnablePassthrough.assign(time=lambda x: time_chain_lcel.invoke({"recipe": x["recipe"]}))
)
result = overall_chain_lcel.invoke({"location": "China"})
```

---

## 6. LangChain Chains, Memory & Agents

### 6.1 Chains (classic style)

A **chain** = a sequence of calls, where each step takes one input and produces one output; the output of step *N* becomes the input of step *N+1*.

**Worked example**: find a famous dish for a location → get its recipe → estimate cooking time.

```python
from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate

# Chain 1: location -> famous dish ("meal")
location_template = """Your job is to come up with a classic dish from the area that the user suggests.
{location}
YOUR RESPONSE: """
location_prompt_template = PromptTemplate(template=location_template, input_variables=['location'])
location_chain = LLMChain(llm=mixtral_llm, prompt=location_prompt_template, output_key='meal')

# Chain 2: meal -> recipe
dish_chain = LLMChain(llm=mixtral_llm, prompt=dish_prompt_template, output_key='recipe')

# Chain 3: recipe -> cooking time
recipe_chain = LLMChain(llm=mixtral_llm, prompt=recipe_prompt_template, output_key='time')

overall_chain = SequentialChain(
    chains=[location_chain, dish_chain, recipe_chain],
    input_variables=['location'],
    output_variables=['meal', 'recipe', 'time'],
    verbose=True
)
result = overall_chain.invoke({'location': 'China'})
# e.g. meal="Peking Duck", recipe=..., time=...
```
Set `verbose=True` to trace how each input transforms through the chain.

### 6.2 Memory

LangChain memory stores conversation history so chains/models retain context across turns:
- A chain **reads from memory** to enrich the current input before running its core logic.
- It then **writes** the current run's inputs/outputs back to memory — preserving continuity across interactions.

```python
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_ai_message("hi!")
history.add_user_message("what is the capital of France?")
history.messages                      # view stored messages
ai_response = mixtral_llm.invoke(history.messages)   # generate a response using history as context
```

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

conversation = ConversationChain(llm=mixtral_llm, verbose=True, memory=ConversationBufferMemory())
response = conversation.invoke(input="Hello, I am a little cat. Who are you?")
```

### 6.3 Agents

An **agent** is a dynamic system where the language model itself **decides and sequences actions** (e.g., choosing which pre-defined chain or tool to run). The model *generates guidance text* but does not directly execute actions — it relies on integrated **tools** (search engines, databases, websites, calculators, etc.) to actually fulfill the request.

Example: user asks for the population of Italy → agent reasons with the LLM → queries a database/tool → returns a curated answer. This shows the agent autonomously combining LLM reasoning with external tools.

**Tools**

```python
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()
python_calculator = Tool(
    name="Python Calculator",
    func=python_repl.run,
    description="Useful for when you need to perform calculations or execute Python code. Input should be valid Python code."
)
result = python_calculator.invoke("a = 3; b = 1; print(a+b)")
```

```python
from langchain.tools import tool

@tool
def search_weather(location: str):
    """Search for the current weather in the specified location."""
    return f"The weather in {location} is currently sunny and 72°F."
```

**ReAct Agents (Reasoning + Acting)**

```python
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm=mixtral_llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
result = agent_executor.invoke({"input": "What is the square root of 256?"})
```

**Pandas DataFrame Agent** — query/visualize tabular data using natural language:

```python
from langchain_experimental.agents import create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(llm=chat_model, df=dataframe, verbose=True)
agent.invoke("how many rows in the data frame")
# The LLM translates the natural-language query into Python code executed behind the scenes.
```

---

## 7. Document Loaders, Splitters, Embeddings & Retrieval (RAG)

### 7.1 The `Document` Object

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Python is an interpreted high-level general-purpose programming language...",
    metadata={
        'my_document_id': 234234,
        'my_document_source': "About Python",
        'my_document_create_time': 1680013019
    }
)
```
- `page_content`: the actual text content.
- `metadata`: arbitrary associated info (id, source, timestamp, etc.).

### 7.2 Document Loaders

```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("path/to/document.pdf")
documents = loader.load()

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
web_data = loader.load()
```

### 7.3 Text Splitters (chunking for embeddings/LLM context limits)

```python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
chunks = text_splitter.split_documents(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```
`RecursiveCharacterTextSplitter` tries each separator in order until chunks reach the target size — generally preferred over the plain `CharacterTextSplitter`.

### 7.4 Embeddings

```python
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}
watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)
```

### 7.5 Vector Stores & Retrievers

```python
from langchain.vectorstores import Chroma

docsearch = Chroma.from_documents(chunks, watsonx_embedding)
docs = docsearch.similarity_search("Langchain")

retriever = docsearch.as_retriever()
docs = retriever.invoke("Langchain")
```

**ParentDocumentRetriever** — embeds small child chunks for accuracy, but returns the larger parent document for full context:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
child_splitter  = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
vectorstore = Chroma(collection_name="split_parents", embedding_function=watsonx_embedding)
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, docstore=store,
    child_splitter=child_splitter, parent_splitter=parent_splitter,
)
retriever.add_documents(documents)
retrieved_docs = retriever.invoke("Langchain")
```

### 7.6 RetrievalQA (a complete RAG chain)

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=mixtral_llm, chain_type="stuff",
    retriever=docsearch.as_retriever(), return_source_documents=False
)
answer = qa.invoke("what is this paper discussing?")
```

---

## 8. Flask Web Framework

### 8.1 Overview

- **Flask** is a **micro-framework** for building web applications in Python — "unopinionated," doesn't force a specific set of tools on you.
- Created by **Armin Ronacher in 2004**, originally as an April Fool's joke; it gained real popularity for its simplicity and extensibility.
- Requires Python ≥ 3.7 (course used Flask 2.2.2).

### 8.2 Main Features

- **Development web server** for running apps locally.
- **Debugger** with interactive tracebacks/stack traces in-browser.
- **Standard Python logging** — usable for custom app log messages too.
- **Testing support** (works with pytest, coverage) enabling test-driven development.
- **Request/response object access** to read arguments and customize responses.

### 8.3 Additional Built-in Capabilities

- **Static assets** (CSS, JS, images) with template tags to load them.
- **Dynamic pages** via the **Jinja** templating engine (content that changes per-request, login checks, etc.).
- **Routing** — supports dynamic URLs (great for RESTful services), multiple HTTP methods per route, and redirection.
- **Global error handlers** at the application level.
- **User session management**.

### 8.4 Popular Community Extensions

| Extension | Purpose |
|---|---|
| **Flask-SQLAlchemy** | ORM support (SQLAlchemy) for working with DB objects in Python |
| **Flask-Mail** | Set up an SMTP mail server |
| **Flask-Admin** | Add admin interfaces easily |
| **Flask-Uploads** | Customized file uploading |
| **Flask-CORS** | Handle Cross-Origin Resource Sharing |
| **Flask-Migrate** | Database migrations for SQLAlchemy ORM |
| **Flask-User** | User authentication/authorization/management |
| **Marshmallow** | Object serialization/deserialization |
| **Celery** | Task queue for background jobs & scheduled/multi-stage workflows |

### 8.5 Installation

```bash
mkdir genai_flask_app
cd genai_flask_app
python3.11 -m venv venv
source venv/bin/activate
pip install flask==2.2.2
```
Pin dependency versions so the app reproduces reliably across dev/staging/production and avoids breakage from silent auto-updates.

**Built-in dependencies** (installed automatically with Flask):

| Package | Role |
|---|---|
| **Werkzeug** | Implements WSGI (Web Server Gateway Interface) — the standard interface between Python apps and servers |
| **Jinja** | Template language that renders your app's pages |
| **MarkupSafe** | Ships with Jinja; escapes untrusted input to prevent injection attacks |
| **ItsDangerous** | Securely signs data (tamper detection); protects Flask session cookies |
| **Click** | Framework for command-line apps; provides the `flask` CLI command and custom management commands |

Use `pip freeze` inside the virtual environment to see all installed built-in packages.

### 8.6 Flask vs. Django

| | **Flask** | **Django** |
|---|---|---|
| Weight | Very light (micro-framework) | Full-stack, "batteries included" |
| Philosophy | Unopinionated, flexible, plug-and-play | Opinionated, makes most decisions for you |
| Dependencies | Minimal by default, extend as needed | Everything included out of the box |

### 8.7 Flask Basics — Routes & Status Codes

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "My first Flask application in action!"
```
- Returning from an `@app.route` handler (or using `jsonify()`) automatically returns HTTP **200 OK** by default.
- You can also return a status explicitly: `return ("My first Flask application in action!", 200)`.

**Common HTTP error codes**

| Code | Meaning |
|---|---|
| 400 | Invalid request (missing/improper parameters) |
| 401 | Missing or invalid credentials |
| 403 | Credentials valid but insufficient permission |
| 404 | Resource not found |
| 405 | Requested operation/method not supported |
| 500 | Server-side error |

```python
@app.route('/')
def search_response():
    query = request.args.get("q")
    if not query:
        return {"error_message": "Input parameter missing"}, 422
    resource = fetch_from_database(query)
    if resource:
        return {"message": resource}
    else:
        return {"error_message": "Resource not found"}, 404

@app.errorhandler(500)
def server_error(error):
    return {"message": "Something went wrong on the server"}, 500
```

---

## 9. Building a GenAI Flask Application (IBM watsonx + LangChain)

### 9.1 Project Setup

```bash
mkdir genai_flask_app
cd genai_flask_app
python3.11 -m venv venv
source venv/bin/activate
pip install ibm-watsonx-ai
```

### 9.2 Authenticate & Configure the Model

```python
from ibm_watsonx_ai import Credentials
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    # api_key="<YOUR_API_KEY>"
)

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}

from ibm_watsonx_ai.foundation_models import ModelInference
model = ModelInference(
    model_id="ibm/granite-3-3-8b-instruct",
    params=params,
    credentials=credentials,
    project_id="skills-network"
)

text = """Only reply with the answer. What is the capital of Canada?"""
print(model.generate(text)['results'][0]['generated_text'])
```

### 9.3 Model-Specific Prompt Templates (e.g., Llama 3 token format)

```python
from langchain.prompts import PromptTemplate

llama3_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "user_prompt"]
)
```

### 9.4 Chaining a Template into the Model

```python
def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model
    return chain.invoke({'system_prompt': system_prompt, 'user_prompt': user_prompt})
```

### 9.5 Structured JSON Output

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 to 100")
    response: str = Field(description="Generated AI response")

json_parser = JsonOutputParser(pydantic_object=AIResponse)

def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    return chain.invoke({
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'format_prompt': json_parser.get_format_instructions()
    })
```

### 9.6 Wrapping It in a Flask API

```python
from flask import Flask, request, jsonify
from model import get_model_response

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    model_name = data.get('model')
    user_message = data.get('message')

    if not user_message or not model_name:
        return jsonify({"error": "Missing message or model selection"}), 400

    system_prompt = "You are an AI assistant helping with customer inquiries. Provide a concise response."
    try:
        response = get_model_response(model_name, system_prompt, user_message)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 10. The GenAI Application Development Lifecycle

Three broad phases developers move through, from proof-of-concept to production:

### 10.1 Ideation & Experimentation

- Your use case is specialized → you likely need a **specialized model**, not necessarily the biggest one.
- Research/evaluate models from hubs like **Hugging Face**; weigh **model size**, **performance**, and **benchmarks**.
- Rules of thumb:
  - **Self-hosting** an LLM is generally cheaper than a cloud-based service.
  - **Small Language Models (SLMs)** often beat LLMs on latency and are more specialized for a single task.
- Experiment with **prompting techniques** early: zero-shot, few-shot, chain-of-thought (see Section 3).
- Test against your own data early to surface limitations before committing further.

### 10.2 Building

- You can **self-host** a model locally (like running a local database) — calling its API from `localhost`, which also keeps data private/secure on-premises.
- To incorporate your own data into an LLM, two common approaches:
  - **RAG (Retrieval-Augmented Generation)** — supplement a pretrained foundation model with relevant retrieved data at inference time (see Section 7).
  - **Fine-tuning** — bake domain data/behavior/style directly into the model's weights.
- Frameworks like **LangChain** simplify building multi-step GenAI apps (chatbots, IT process automation, data management) by chaining sequences of prompts and model calls, and by helping evaluate the resulting flows.

### 10.3 Development & Operations (MLOps)

- Deploying to production requires **efficient model deployment & scaling**: containers, orchestrators (**Kubernetes**), auto-scaling/load-balancing, production-grade serving runtimes (e.g., **vLLM**).
- Many orgs adopt a **hybrid, multi-model approach**: different models for different use cases, combined with a mix of on-prem and cloud infrastructure to optimize cost/resources.
- Ongoing responsibilities: **benchmark, monitor, and handle exceptions** in production — this is the **MLOps** discipline (the ML analogue of DevOps), ensuring models run smoothly and stay current over time.

---

## 11. Choosing the Right Model for the Right Use Case

### 11.1 The "Garden" Analogy

Like growing a variety of vegetables (not living on carrots alone), a **multi-model approach** means picking different models suited to different use cases rather than forcing one model to do everything.

### 11.2 Key Questions When Evaluating a Model

- Who built it?
- What data was it trained on?
- What guardrails are in place?
- What risks/regulations must be considered?

### 11.3 The Selection Process

1. **Write a specific prompt** that captures: the use case, the user's problem, the ask of the technology, and the guardrails for "good" output.
2. **Research available models**: size, performance, cost, risk, deployment method.
3. **Evaluate models against the prompt** — start with a large model, get it to satisfy the prompt, then try to reproduce the result with smaller models to find the best cost/performance fit.
4. **Continually evaluate and govern** the chosen model — ongoing testing against performance/cost benchmarks, updating data/prompts as needed, and staying open to swapping in newer models (avoid vendor/model lock-in).

### 11.4 Factors Affecting Model Choice

Performance (accuracy, reliability, speed), size, deployment method, transparency, and potential risks.

### 11.5 Team & Governance

Implementation should be a **cross-disciplinary, cross-departmental** collaborative effort — not owned by a single team. The team should be able to diagnose performance benchmarks and make informed, data-driven decisions about current and future models.

---

## 12. Quick-Reference Cheat Sheets

### 12.1 Foundations of Generative AI & LangChain — Setup & Prompting

| Method | Description |
|---|---|
| `pip install "ibm-watsonx-ai==1.0.8" "langchain==0.2.11" "langchain-ibm==0.1.7" "langchain-core==0.2.43"` | Install course dependencies |
| `warnings.filterwarnings('ignore')` | Suppress noisy warnings |
| `WatsonxLLM(model_id=..., url=..., project_id=..., params=...)` | Interact with IBM watsonx LLMs |
| `GenParams` (`ibm_watsonx_ai.metanames.GenTextParamsMetaNames`) | Controls generation params: `MAX_NEW_TOKENS`, `MIN_NEW_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K` |
| Basic / Zero-shot / One-shot / Few-shot / CoT / Self-consistency prompts | See Section 3.4 |
| `PromptTemplate`, `RunnableLambda`, `StrOutputParser`, LCEL `|` pattern | See Sections 4–5 |

### 12.2 Introduction to LangChain — Core Classes

`WatsonxLLM`, message types (`SystemMessage`/`HumanMessage`/`AIMessage`), `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, `JsonOutputParser`, `CommaSeparatedListOutputParser`, `Document`, `PyPDFLoader`, `WebBaseLoader`, `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `WatsonxEmbeddings`, `Chroma`, retrievers, `ParentDocumentRetriever`, `RetrievalQA`, `ChatMessageHistory`, `ConversationBufferMemory`, `LLMChain`, `SequentialChain`, `RunnablePassthrough`, `Tool`, `@tool`, `create_react_agent`, `AgentExecutor` — **full code samples in Sections 4, 6, 7.**

### 12.3 Build a GenAI Application with LangChain (Flask + watsonx)

Project scaffolding → virtual env → `ibm-watsonx-ai` install → `Credentials` → `GenTextParamsMetaNames` → `ModelInference` → prompt generation → LangChain prompt templates (model-specific token formats, e.g. Llama 3) → LCEL chaining (`template | model`) → `JsonOutputParser` for structured output → Flask `/generate` POST endpoint. **Full code in Section 9.**

### 12.4 Web Development Using Flask

`Flask(__name__)`, `@app.route`, default 200 OK, HTTP error codes (400/401/403/404/405/500), `@app.errorhandler`. **Full code in Section 8.7.**

---

## 13. Retrieval-Augmented Generation (RAG) Fundamentals

### 13.1 The Problem RAG Solves

LLMs ("the generation part") answer confidently from what they memorized during training. This creates two classic **LLM challenges**:
1. **No source** — the model can't cite where an answer came from, so it may **hallucinate** (make up plausible-sounding but false information).
2. **Out of date** — the model's knowledge is frozen at its training cutoff; retraining it every time new information appears is impractical.

**Illustrative anecdote**: asked "which planet has the most moons," an LLM might confidently answer "Jupiter" from stale training data, when the up-to-date answer (from a live source like NASA) is "Saturn" — because new moons keep being discovered.

### 13.2 What RAG Adds

RAG adds a **content store** (open, like the public internet, or closed, like a private document collection) that the LLM consults *before* answering. The prompt sent to the LLM is no longer just the user's question — it now has **three parts**:
1. An instruction to pay attention to retrieved content,
2. The retrieved content itself,
3. The user's original question.

Only then does the model generate its answer — and it can now **cite evidence** for that answer.

### 13.3 How RAG Fixes the Two Challenges

- **Out-of-date fix**: instead of retraining the whole model, you simply **update the data store** with new/current information; the next query automatically retrieves the freshest facts.
- **No-source fix**: the model is instructed to ground its answer in retrieved primary-source data, which makes it **less likely to hallucinate** and lets it **show evidence**. It also enables a valuable behavior: the model can say **"I don't know"** when the data store can't answer the question, instead of confidently making something up.
- **Trade-off**: if the retriever isn't good enough to surface high-quality grounding information, an otherwise-answerable question may still fail to get answered — so both the **retriever** and the **generative** side need continual improvement.

### 13.4 The 8-Step RAG Pipeline

*(See the accompanying RAG-Process diagram: Sources → Embed Sources → Vector Store → Retriever → Retrieved Text → combined Prompt → LLM → Response.)*

1. **Load and chunk** the source documents.
2. **Embed** the chunks/documents — convert text into numerical vectors using an embedding model.
3. **Store the vectors**, typically in a vector database.
4. **Accept** the user's original prompt/query.
5. **Embed the user's prompt** using the *same* embedding model used for the source documents.
6. **Retrieve** — use a retriever to compare the prompt embedding against the stored document embeddings and pull back the most similar chunks.
7. **Augment** the user's prompt with the retrieved information.
8. **Pass the augmented prompt to the LLM** to produce a context-aware, grounded response.

---

## 14. LlamaIndex

### 14.1 Overview

**LlamaIndex** is a framework for building **LLM-powered context augmentation** — making your own data available to an LLM so it can ground its responses in that data. Typical use cases:
- **Question answering with RAG** (its primary focus).
- **Chatbots** — extend basic RAG with multi-turn back-and-forth, clarifying/follow-up questions.
- **Document understanding & data extraction** — semantically pulling out names, dates, addresses, figures, etc. from structured or unstructured data.

### 14.2 The `Document` Class

LlamaIndex's generic container for a document source.

```python
from llama_index.core import Document
doc = Document(text="Hello LlamaIndex")
doc.dict()   # inspect stored items
```

A `Document` stores:
- **id_** — a unique identifier,
- **embedding** — a placeholder if you embed the whole document,
- **metadata** — a dict (origin, creation date, etc.),
- **relationships** — a dict linking to other related documents,
- **text** — the document's actual text content.

Supported formats: text, PDF, Markdown, CSV, JSON, HTML — plus numerous connectors for databases and cloud storage.

### 14.3 Loading Documents — `SimpleDirectoryReader`

```python
from llama_index.core import SimpleDirectoryReader

# Load every file in a directory
documents = SimpleDirectoryReader("path/to/dir").load_data()

# Recursively include subdirectories
documents = SimpleDirectoryReader("path/to/dir", recursive=True).load_data()

# Load only specific files
documents = SimpleDirectoryReader(input_files=["a.pdf", "b.md"]).load_data()

# Load only specific file types
documents = SimpleDirectoryReader("path/to/dir", required_exts=[".pdf", ".md"]).load_data()
```
Returns a list of `Document` objects. For connectors beyond the core library (databases, RSS feeds, cloud storage, etc.), LlamaIndex maintains a registry called **LlamaHub** (e.g., `DatabaseReader` for SQL queries, `JSONReader`, `RssReader`).

### 14.4 Chunking — Nodes & `SentenceSplitter`

A **LlamaIndex node** = a text chunk (structurally similar to a `Document`, also carrying metadata/relationships/embeddings).

```python
from llama_index.core.node_parser import SentenceSplitter

node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)
nodes = node_parser.get_nodes_from_documents(documents)
```
- `chunk_size` — max chunk size in tokens.
- `chunk_overlap` — max tokens allowed to overlap between consecutive chunks.
- `SentenceSplitter` recursively splits on separators like newlines and periods, keeping each chunk under the token limit.

Other splitters:
- **Semantic Splitter** — splits wherever sentence-to-sentence similarity drops below a threshold.
- **`LangChainNodeParser`** — a wrapper letting you use *any* LangChain text splitter inside LlamaIndex.

### 14.5 Embeddings & Storage — `VectorStoreIndex`

**Simple case** (default embedding model, in-memory storage):
```python
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex(nodes)
```

**Custom embedding model + persistent storage** (e.g., ChromaDB + a HuggingFace embedding model):
```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, VectorStoreIndex

embed_model = HuggingFaceEmbedding(model_name="...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes,
    embed_model=embed_model,
    storage_context=storage_context
)
```
`VectorStoreIndex` handles both **embedding generation** and **storage** in one class — whether in-memory or backed by an external vector database (Chroma DB, FAISS, Milvus, etc.).

### 14.6 Retrieval

```python
retriever = index.as_retriever(similarity_top_k=5)
results = retriever.retrieve(user_prompt)
```
- `as_retriever()` creates a retriever tied to the index (guaranteeing the *same* embedding model is used for both the stored nodes and the incoming prompt).
- `similarity_top_k` controls how many of the most similar nodes are returned (default example: 5), ranked most-similar first.

### 14.7 Response Synthesizer & Query Engine

**Response synthesizer** — combines *prompt augmentation + LLM querying + response generation* into one step, given an already-retrieved set of nodes:
```python
response = response_synthesizer.synthesize(user_prompt, nodes=retrieved_nodes)
```
It embeds the prompt, augments it with the retrieved nodes, and queries the LLM — all internally.

**Query engine** — compresses the *entire* RAG pipeline (embedding, retrieval, augmentation, querying, response generation) into a single object:
```python
query_engine = index.as_query_engine()
response = query_engine.query(user_prompt)
```
This dramatically reduces the amount of RAG boilerplate code needed.

**Customization options** for a query engine:
- Swap the default **LLM** for a different one.
- Define a **custom prompt template** for augmentation.
- Specify a **custom retriever**.

---

## 15. LangChain vs. LlamaIndex — Detailed Comparison

Both are frameworks for **LLM-powered context augmentation**, commonly used for RAG apps, chatbots, document understanding, and summarization/extraction. The comparison below follows the 8-step RAG pipeline from Section 13.4.

### 15.1 Step 1 — Loading Source Documents

| | LangChain | LlamaIndex |
|---|---|---|
| Core loaders | `TextLoader`, `CSVLoader`, `JSONLoader`, `WebBaseLoader` (uses BeautifulSoup4), `DoclingLoader` (PDF/DOCX/PPTX/HTML via Docling), `UnstructuredLoader` (via the `unstructured` library) | `SimpleDirectoryReader` — natively handles markdown, PDF, Word, PowerPoint, and more |
| Whole directories | `DirectoryLoader` (defaults to `UnstructuredLoader` internally, swappable) | `SimpleDirectoryReader` — supports recursive traversal and extension filters natively |
| Extended connectors | Many integrations: SQL databases, AWS S3, Figma files, etc. | **LlamaHub** registry — e.g. `DatabaseReader` (SQL), `JSONReader`, `RssReader` |
| Design philosophy | Modular, relies heavily on external integrations/packages | More self-contained "native" solutions; falls back to external libraries only when needed |

**Takeaway**: LlamaIndex's `SimpleDirectoryReader` gives a stronger out-of-the-box experience across many file types; LangChain's `DirectoryLoader` is more configurable/flexible since it can be pointed at any of LangChain's many loaders.

### 15.2 Step 1 (cont.) — Chunking Documents

**LangChain splitters**
- **`CharacterTextSplitter`** — splits on a character sequence (e.g. `\n\n`), size capped by max character length (or by token count in token mode).
- **`TokenTextSplitter`** — encodes to tokens, splits by token-length, then decodes back to text.
- **`RecursiveCharacterTextSplitter`** — tries a list of separators in order (e.g. `["\n\n", "."]`); if a chunk is still too long, it recursively splits on the next separator in the list.
- **Document-structured splitters** — e.g. `MarkdownHeaderTextSplitter` (splits on `#`, `##`, etc.); also available for code, HTML, and a recursive JSON splitter.
- **`SemanticChunker`** — splits where sentence-to-sentence similarity drops below a threshold (finds natural conceptual breaks).

**LlamaIndex splitters** (chunks are called **nodes**)
- **`SentenceSplitter`** — similar to `RecursiveCharacterTextSplitter` but token-based (chunk size defined in tokens); ships with a more comprehensive default separator list.
- File-based **node parsers** for HTML, JSON, and Markdown (LlamaIndex's counterpart to LangChain's document-structured splitters).
- A **code splitter** — text-based rather than file-based (unlike LangChain's).
- **`SemanticSplitterNodeParser`** — LlamaIndex's counterpart to `SemanticChunker`.
- **`LangChainNodeParser`** — wraps *any* LangChain splitter for use inside LlamaIndex.

**Takeaway**: both frameworks offer comprehensive splitting strategies; LlamaIndex's default `SentenceSplitter` setup tends to be a bit more thorough out of the box, but LlamaIndex also lets you borrow any LangChain splitter if preferred.

### 15.3 Step 2 — Embedding

- Both frameworks support many embedding integrations (HuggingFace, OpenAI, etc.).
- LlamaIndex provides a wrapper so you can use any **LangChain-compatible embedding model** inside LlamaIndex.
- **Key design difference**: in **LlamaIndex**, embeddings are typically generated *and* stored in the vector store in **one single step**. In **LangChain**, embeddings are generated first, then stored as a **separate step**.

### 15.4 Step 3 — Storing Vectors

| | LangChain | LlamaIndex |
|---|---|---|
| Built-in store | `InMemoryVectorStore` (only in-memory option defined in core) | `VectorStoreIndex` — in-memory by default |
| External DB integrations | `Chroma`, `FAISS`, `Milvus`, `PGVector` (pgvector-extended PostgreSQL) | Wraps external DBs like Chroma DB or FAISS **inside the same `VectorStoreIndex` class** — swap backend without changing downstream code |
| One-line embed+store | Not typical (two steps) | `index = VectorStoreIndex(nodes)` |
| Metadata | Often requires **manual metadata setup**; handling varies by backend | Chunk metadata is **automatically created and stored** |
| Flexibility | No single unifying class → more granular control over each vector store's unique features | Unifying class → simpler, more uniform usage |

### 15.5 Step 4 — Accepting the User's Prompt

Neither framework has a specific implementation here — both simply assume a prompt is handed to them from an external workflow/process.

### 15.6 Step 5 — Embedding the User's Prompt

Both frameworks typically combine this with retrieval, embedding the prompt via a retriever built from the vector store object — ensuring the **same embedding model** is used for both the prompt and the stored chunks. No notable implementation differences here.

### 15.7 Step 6 — Retrieving Relevant Chunks

- Basic top-k similarity retrieval works similarly in both.
- Both offer advanced retrieval patterns. Example: LangChain's **Parent Document Retriever** retrieves the *parent* document containing a relevant chunk (rather than just the chunk) — this pattern needs **two vector stores** (one for chunks, one for parent documents).
- Recommendation: if your project needs a specific retrieval pattern, investigate each framework's retriever options directly to pick the best fit.

### 15.8 Step 7 — Augmenting the Prompt

Both use **prompt templates** with placeholders for the original prompt + retrieved text, but combine this step differently:

- **LangChain**: prompt augmentation is a **standalone step**, not merged with anything before/after it → easy to customize templates.
- **LlamaIndex**: augmentation is **merged with LLM response generation** (via a response synthesizer), or merged with *everything* (embedding + retrieval + augmentation + generation) via a query engine. Default templates work well for most cases, but customizing them is a bit more involved since the step is bundled with others.

### 15.9 Step 8 — Passing the Augmented Prompt to the LLM

- **LangChain**: done **manually** — e.g. `response = llm.invoke(messages)`.
- **LlamaIndex**: bundled into the augmentation step via either:
  - a **response synthesizer** (takes prompt + retrieved nodes → internally augments → returns LLM response), or
  - a **query engine** (takes just the original prompt → internally does embedding, retrieval, augmentation, and generation → returns the response).

### 15.10 Overall Takeaways

| | LangChain | LlamaIndex |
|---|---|---|
| Strengths | Numerous integrations, modular design, easy component-level customization | Simplicity, ease of development, strong native/sensible defaults |
| Weaknesses | Often needs manual setup (e.g. metadata) | Customization is a bit harder since steps are bundled together |
| Best for | Projects needing deep integration flexibility & granular control | Projects wanting to get a solid RAG pipeline running quickly with minimal boilerplate |

Both frameworks are fully capable of handling typical RAG workflows — the right choice depends on how much granular control vs. out-of-the-box simplicity your project needs.

---

## 16. Gradio — Building Web UIs for Models

### 16.1 What Is Gradio?

**Gradio** is an open-source Python library/package for quickly building **customizable web-based user interfaces** for Python functions — especially handy for machine learning models and computational tools. **No JavaScript, CSS, or web-hosting experience required.**

### 16.2 The Core Workflow

1. Write the Python function(s)/logic for your application.
2. Create a Gradio **interface** for those functions, specifying inputs/outputs.
3. Configure how users will interact with the app.
4. **Launch** the Gradio server (`launch()`), which starts a local web server.
5. Access the interface via the **local or public URL** Gradio provides; users interact with it in real time.

### 16.3 Installation & a Minimal Example

```bash
pip install gradio
```

```python
import gradio as gr

def echo_text(text):
    return text

demo = gr.Interface(
    fn=echo_text,
    inputs=gr.Textbox(label="Enter text"),
    outputs=gr.Textbox(label="Output")
)
demo.launch()
```

### 16.4 The `Interface` Class — Core Arguments

| Argument | Description |
|---|---|
| **`fn`** | The Python function to wrap. Each function parameter maps to one input component; the function's return value should be a single value (one output component) or a tuple (matching multiple output components one-to-one). |
| **`inputs`** | The Gradio component(s) used for inputs — count must match the function's argument count. |
| **`outputs`** | The Gradio component(s) used for outputs. |

### 16.5 Multiple Inputs Example

```python
import gradio as gr

def process(text, number):
    return f"{text} — {number}"

demo = gr.Interface(
    fn=process,
    inputs=[gr.Textbox(label="Name"), gr.Number(label="Age")],
    outputs=gr.Textbox()
)
demo.launch()
```

### 16.6 File Uploads

```python
import gradio as gr

def count_files(files):
    return len(files)

demo = gr.Interface(fn=count_files, inputs=gr.File(file_count="multiple"), outputs=gr.Number())
demo.launch()
```
`gr.File` supports multiple uploads and provides file paths for backend processing.

### 16.7 Common Input Component Types

| Component | Description |
|---|---|
| **`checkbox`** | Boolean toggle — `True`/`False` only |
| **`checkboxGroup`** | Select multiple values from a predefined checkbox list |
| **`dropdown`** | Dropdown list; one value by default, or multiple if `multiselect=True` |
| **`file`** | Upload a file |
| **`image`** | Select/upload an image |
| **`radio`** | Force the user to choose exactly one value |
| **`slider`** | Pick a value in a min–max range; `value` sets the default, `step` sets the increment (use integers for integer-only selection) |
| **`textbox`** | Expandable free-text input |

### 16.8 Common Output Component Types

| Component | Description |
|---|---|
| **`textbox`** | Expandable text output |
| **`label`** | Typically for classification models — shows classes with predicted probabilities; `num_top_classes` controls how many classes are shown |

### 16.9 Launching the Application

```python
demo.launch(share=True)
```
- `launch()` starts a simple local web server serving the interface.
- Setting **`share=True`** additionally creates a **public URL**, so anyone worldwide can access and interact with the app — while all computation still runs on your local machine.

---

## 17. Course 2 Cheat Sheets & Appendix

### 17.1 Cheat Sheet: Build RAG Apps with LlamaIndex

- LlamaIndex's `Document` class wraps/stores entire documents, including embeddings, metadata (creation time, source directory, etc.), and relationships to other documents.
- Documents are loaded via `SimpleDocumentReader` (individual files or whole directories); more loaders/connectors are available at **LlamaHub.ai**.
- Documents are chunked into **Nodes** via document splitters. Nodes mirror `Document`'s structure (metadata, relationships, embeddings).
- `SentenceSplitter` recursively splits on a character list while capping chunk size by token count; `LangChainNodeParser` wraps any LangChain text splitter for use in LlamaIndex.
- Embeddings are generated **and** stored in one step via `VectorStoreIndex`, which stores in-memory by default but can wrap Chroma DB, FAISS, or Milvus.
- A retriever (`index.as_retriever()`) embeds the user's prompt and retrieves relevant chunks in one step, guaranteeing the same embedding model is used throughout.
- Prompt augmentation + LLM querying happen together — either via a **response synthesizer** (input: prompt + already-retrieved nodes) or a **query engine** (`index.as_query_engine()`, input: just the raw prompt).
- Prompt augmentation itself is controlled by customizable prompt templates with placeholders for the prompt and retrieved chunks.
- **LlamaIndex** strengths: simplicity, ease of development, powerful native tools. **LangChain** strengths (by comparison): more external integrations, more modular design, easier per-component customization.

### 17.2 Cheat Sheet: Building Apps with RAG (Gradio)

- Gradio's `Interface` class wraps any Python function; core args are `fn`, `inputs`, `outputs` (see Section 16.4).
- Common input types: `checkbox`, `checkboxGroup`, `dropdown`, `file`, `image`, `radio`, `slider`, `textbox` (Section 16.7).
- Common output types: `textbox`, `label` (with `num_top_classes` for classification, Section 16.8).
- `launch()` starts a local web server; `share=True` exposes a public URL while computation stays on your machine.

### 17.3 Reading: LangChain vs. LlamaIndex

A full step-by-step comparison across the 8-stage RAG pipeline (loading, chunking, embedding, storage, prompt acceptance, prompt embedding, retrieval, augmentation, and LLM querying) — reproduced in Section 15 above.

### 17.4 RAG Process Diagram (Reference)

The 8-step pipeline diagram referenced throughout Sections 13–15:
`Sources → ① Embed Sources → ② Vector Store → ③` while in parallel `Prompt → ④ Embed Prompt → ⑤`; both feed the `⑥ Retriever → Retrieved Text ⑦`, which combines with the original `Prompt` and goes into the `⑧ LLM → Response`.

---

## Appendix: Source Material Map

**Course 1** (Generative AI, Prompt Engineering, LangChain Foundations, Flask):
- Video lessons: LangChain Core Concepts; LCEL Chaining Method; Advanced Prompt Engineering; Introduction to LangChain; In-Context Learning; Introduction to Generative AI; Foundation Models overview (IBM Research); Introduction to NLP; LangChain Chains & Agents; Introduction to Flask; Building GenAI Applications (developer journey); Choosing the Right AI Model.
- Cheat sheets: *Build GenAI Application with LangChain*, *Web Development Using Flask*, *Introduction to LangChain in GenAI Applications*, *Foundations of Generative AI and LangChain*.
- Supporting files: `Build_Smarter_AI_Apps_Empower_LLMs_with_LangChain.ipynb`, `genai_flask_app.tar`, `lab-instructions.md`.

**Course 2** (RAG, LlamaIndex, LangChain vs. LlamaIndex, Gradio):
- Video lessons: Retrieval Augmented Generation overview (IBM Research); LlamaIndex document ingestion & chunking; LlamaIndex vector stores & query engines; Getting Started with Gradio.
- Readings/cheat sheets: *LangChain vs LlamaIndex*; *Cheatsheet: Build RAG Apps with LlamaIndex*; *Cheat Sheet: Building Apps with RAG* (Gradio).
- Supporting files: `Summarize_private_documents_using_RAG_LangChain_and_LLMs.ipynb`, `icebreaker.tar`, `project.tar`, `lab-instructions (1/2/3).md`, RAG-Process diagram image.

