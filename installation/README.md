# Installation

## Official release

To install LangChain run:

### Pip

```sh
pip install langchain
```

### Conda

```sh
conda install langchain -c conda-forge
```

This will install the bare minimum requirements of LangChain. A lot of
the value of LangChain comes when integrating it with various model
providers, datastores, etc. By default, the dependencies needed to do
that are NOT installed. You will need to install the dependencies for
specific integrations separately.

## From source

If you want to install from source, you can do so by cloning the repo
and be sure that the directory is
`PATH/TO/REPO/langchain/libs/langchain` running:

```sh
pip install -e .
```

## LangChain core

The `langchain-core` package contains base abstractions that the rest of
the LangChain ecosystem uses, along with the LangChain Expression
Language. It is automatically installed by `langchain`, but can also be
used separately. Install with:

```sh
pip install langchain-core
```

## LangChain community

The `langchain-community` package contains third-party integrations. It
is automatically installed by `langchain`, but can also be used
separately. Install with:

```sh
pip install langchain-community
```

## LangChain experimental

The `langchain-experimental` package holds experimental LangChain code,
intended for research and experimental uses. Install with:

```sh
pip install langchain-experimental
```

## LangGraph

`langgraph` is a library for building stateful, multi-actor applications
with LLMs, built on top of (and intended to be used with) LangChain.
Install with:

```sh
pip install langgraph
```

## LangServe

LangServe helps developers deploy LangChain runnables and chains as a
REST API. LangServe is automatically installed by LangChain CLI. If not
using LangChain CLI, install with:

```sh
pip install "langserve[all]"
```

for both client and server dependencies. Or
`pip install "langserve[client]"` for client code, and
`pip install "langserve[server]"` for server code.

## LangChain CLI

The LangChain CLI is useful for working with LangChain templates and
other LangServe projects. Install with:

```sh
pip install langchain-cli
```

## LangSmith SDK

The LangSmith SDK is automatically installed by LangChain. If not using
LangChain, install with:

```sh
pip install langsmith
```
