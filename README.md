# Flight Reservation System with LLM Integration

This project provides an interactive flight reservation system powered by Language Models (LLMs) via the OpenAI API. It features capabilities for finding available flights and booking them based on user input through a Gradio interface. It includes components for testing and benchmarking the system’s accuracy and performance.

## Project Overview

The system allows a user to:
1. **Find Flights** between two cities on a specific date.
2. **Book Flights** by flight ID.

The responses are managed through JSON, with specific LLM prompts guiding the system to respond as a travel agent. The agent has access to LLMs for response generation, with a structure for handling user conversation and program state.

## Features

- **LLM-powered responses** using OpenAI’s API.
- **Flight search and booking** functionalities with custom responses in JSON format.
- **Gradio Interface** for a user-friendly chat experience.
- **Benchmarking and evaluation module** to test the system against a predefined dataset.

## Requirements

- Python 3.7+
- Required Python packages:
  - `openai`
  - `datasets`
  - `dataclasses`
  - `tqdm`
  - `gradio`
  - `yaml`

## Project Structure

- `Agent`: The core class responsible for handling user requests and managing conversations.
- `Flight`: Dataclass to represent flight details.
- `AgentResponse` and subclasses: To manage different types of responses from the agent.
- `find_flights` and `book_flight`: Functions for searching and booking flights.
- `eval_agent`: Benchmarking function to evaluate the agent’s accuracy.
- `Gradio Interface`: Interface for user interaction.
- `benchmark`: Run the evaluation against the dataset.

## Usage

### Setting Up the API

Ensure you have the OpenAI API key configured in `client_json` and `client_natural` objects.

```python
client_json = OpenAI(base_url="http://199.94.61.113:8000/v1/", api_key="")
client_natural = OpenAI(base_url="http://199.94.61.113:8000/v1/", api_key="")
