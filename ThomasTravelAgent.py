
from openai import OpenAI
from typing import Dict, List, Optional
from datasets import load_dataset
from datetime import date, time, datetime
import json
import dataclasses
import yaml
import re
import gradio as gr
from tqdm import tqdm

client_json = OpenAI(base_url="http://199.94.61.113:8000/v1/", api_key="")
client_natural = OpenAI(base_url="http://199.94.61.113:8000/v1/", api_key="")

@dataclasses.dataclass
class Flight:
    id: int 
    date: date
    airline: str
    flight_number: str
    origin: str
    destination: str
    departure_time: time
    arrival_time: time
    available_seats: int

def parse_flight(flight):
    return Flight(
    id=flight["id"], 
    date=datetime.strptime(flight["date"], "%Y-%m-%d").date(),
    airline=flight["airline"],
    flight_number=flight["flight_number"],
    origin=flight["origin"],
    destination=flight["destination"],
    departure_time=datetime.strptime(flight["departure_time"], "%H:%M").time(),
    arrival_time=datetime.strptime(flight["arrival_time"], "%H:%M").time(),
    available_seats=flight["available_seats"],
    )

def load_flights_dataset() -> List[Flight]:
    return [
        parse_flight(flight)
        for flight in load_dataset("nuprl/llm-systems-flights", split="train")
    ]

@dataclasses.dataclass
class AgentResponse:
    text: str

@dataclasses.dataclass
class FindFlightsResponse(AgentResponse):
    available_flights: List[int]

@dataclasses.dataclass
class BookFlightResponse(AgentResponse):
    booked_flight: Optional[int] 

@dataclasses.dataclass
class TextResponse(AgentResponse):
    pass

class Agent:
    def __init__(self, client_json: OpenAI, client_natural: OpenAI, flights: List[Flight]):
        self.conversation = []
        self.text_prefix = None
        self.flights = flights
        self.client_json = client_json
        self.client_natural = client_natural
        self.program_state: Dict[str, Optional[int]] = {}
        self.use_llm2 = False 
        self.storedFlights: {object, object}
        self.flight_ids = []
        self.isDone = False

        self.conversation.append(
            {"role": "system", 
            "content": """
                You are a travel agent with access to the following tools and can respond **ONLY** JSON format, specifying the action, origin, destination, date, flight_id, etc.:
                1. Find Flights: Search for flights between two cities on a specific date.
                    The date should have the year as 2023
                    Input format: {"action": "find-flights", "origin": <origin>, "destination": <destination>, "date": <date>}
                2. Book Flight: Book a flight by its flight ID.
                    Input format: {"action": "book-flight", "flight_id": <flight_id>}

                **ONLY** provide integers when a user asks you to book a flight, and **ONLY** use the ids from self.flight_ids. Also, lets say the user asks you to book the nth one, choose the nth element from self.flight_ids if it exists.
                **DO NOT** assume any information that is not provided. Use Any for placeholder.
                **DO NOT** return any other text or natural language. Only return JSON.
            """}
        )

    def find_flights(self, origin: str, destination: str, flight_date: date) -> FindFlightsResponse: 
        origin_code = origin.split("(")[-1].strip(")").strip()
        destination_code = destination.split("(")[-1].strip(")").strip()

        print(origin_code)
        print(destination_code)
        result = [flight for flight in self.flights if flight.origin == origin_code and flight.destination == destination_code and flight.date == flight_date]

        if(origin_code != "Any" and destination_code == "Any"):
            result = [flight for flight in self.flights if flight.origin == origin_code and flight.date == flight_date]
        elif(origin_code == 'Any' and destination_code != 'Any'):
            result = [flight for flight in self.flights if flight.destination == destination_code and flight.date == flight_date]
        
        self.program_state["last_action"] = "find_flights"
        self.program_state["available_flights"] = result

        flight_details = [{"id": flight.id, "airline": flight.airline, "flight_number": flight.flight_number} for flight in result]

        self.conversation.append({
            "role": "assistant",
            "content": f"Available flights: {flight_details}"
        })
        return result
    
    def book_flight(self, flight_id: int) -> BookFlightResponse:
        for flight in self.flights:
            if flight.id == flight_id and flight.available_seats > 0:
                flight.available_seats -= 1
                self.program_state["last_action"] = "book_flight"
                self.program_state["booked_flight"] = flight.id
                self.use_llm2 = True
                return flight.id
        return None
    
    def LLM_Call1(self):
        response = self.client_json.chat.completions.create(
            messages=self.conversation,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0
        )
        llm_response = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": llm_response})
        return llm_response

    def LLM_Call2(self):
        response = self.client_natural.chat.completions.create(
            messages=self.conversation,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0
        )
        llm_response = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": llm_response})
        return llm_response

    def extract_json_from_response(self, response: str) -> Optional[dict]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_matches = re.findall(r'{.*?}', response, re.DOTALL)
        if json_matches:
            try:
                return json.loads(json_matches[0])
            except json.JSONDecodeError:
                pass
        return None
    
    def say(self, user_message: str) -> AgentResponse: 
        self.conversation.append({"role": "user", "content": user_message})

        if(self.isDone):
            self.conversation.append({"role": "system", "content": "Consider the task finished, and if user says anything, respond with something that indicates that you have helped them. Additionally, responses no longer need to be in json format. If the user expresses an issue with the booking, tell them to refresh the page to start a new session"})
            self.conversation.append({"role": "user", "content": user_message})

        if(self.use_llm2):
            llm_response = self.LLM_Call2()
            self.use_llm2 = False
            return TextResponse(text=llm_response)
        else:
            llm_response = self.LLM_Call1()
        
        parsed = self.extract_json_from_response(llm_response)
        if not parsed:
            return TextResponse(text=f"{llm_response}")
        action = parsed.get('action', '')

        if action == 'find-flights':
            origin = parsed.get("origin", "")
            destination = parsed.get("destination", "")
            flight_date = date.fromisoformat(parsed.get("date", ""))
            given_flights = self.find_flights(origin, destination, flight_date)
            
            available_flights_ids = []
            parsed = {
                'action': 'find-flights', 
                'origin': origin, 
                'destination': destination, 
                'date': flight_date, 
                'flights': []
            }
            for f in given_flights: 
                PossibleFlight = {
                    'flight_id': -1, 
                    'airline': "", 
                    'departure_time': "", 
                    'arrival_time': ""
                }
                available_flights_ids.append(f.id)
                self.flight_ids.append(f.id)
                PossibleFlight.update({"flight_id": f.id})
                PossibleFlight.update({"airline": f.airline})
                PossibleFlight.update({"departure_time": f.departure_time})
                PossibleFlight.update({"arrival_time": f.arrival_time})
                parsed['flights'].append(PossibleFlight)
            
            self.storedFlights = parsed
            self.program_state["available_flights"] = available_flights_ids
            if len(available_flights_ids) == 0: 
                return FindFlightsResponse(text="No flights found with the provided information. Please restart window and try a different set of inputs.", available_flights=available_flights_ids)
            return FindFlightsResponse(text=llm_response, available_flights=available_flights_ids)
        elif action == 'book-flight':
            print("PARSED")
            print(parsed)
            print("STORED")
            print(self.storedFlights)
            print("IDS")
            print(self.flight_ids)
            flight_id = parsed.get('flight_id')
            
            if(len(self.storedFlights["flights"]) == 0):
                flight_id = None
                print("id is none.")
            
            if flight_id is not None:
                flight_id = int(flight_id)
                available_flights = self.program_state.get("available_flights", [])
                if flight_id and flight_id in available_flights:
                    booked_flight = self.book_flight(flight_id)
                    self.isDone = True
                    return BookFlightResponse(text=llm_response, booked_flight=booked_flight)
                else:
                    self.isDone = True
                    return BookFlightResponse(text="Invalid flight ID or flight not available.", booked_flight=None)
            else: 
                self.conversation.append({"role": "system", "content": "Consider the attempt to book the flight as failed. If the user wishes to book a flight, tell them to refresh the window to start a new session. Also responses no longer need to be in json format. "})
                return BookFlightResponse(text="Given flight is not available", booked_flight=None)
        else:
            return TextResponse(text=llm_response)
        
    def agent_gui(self, message, history): 
        if len(history) == 0:
            response_1 = self.say(message)
            if(not hasattr(response_1, 'available_flights')):
                return response_1.text
            return str(response_1.available_flights)      
        elif self.program_state["last_action"] == "find_flights":
            response_2 = self.say(message)
            if(not hasattr(response_2, 'booked_flight') or (hasattr(response_2, 'booked_flight') and response_2.booked_flight == None)):
                return response_2.text
            return str(response_2.booked_flight)
        else: 
            response_3 = self.say(message)
            return response_3.text
            

@dataclasses.dataclass
class EvaluationResult:
    score: float
    conversation: List[dict]
    errors: List[str] 

def eval_agent(client_json, client_natural, benchmark_file: str, flights: List[Flight]) -> EvaluationResult:
    agent = Agent(client_json=client_json, client_natural=client_natural, flights=flights)

    with open(benchmark_file, "r") as file:
        steps = yaml.safe_load(file)

    errors = []
    num_tests = len(steps)
    correct_tests = 0

    for n, step in tqdm(enumerate(steps)):
        print(step["prompt"])
        agent.conversation = []
        agent.conversation.append(
            {"role": "system", 
            "content": """
                You are a travel agent with access to the following tools and can respond **ONLY** JSON format, specifying the action, origin, destination, date, flight_id, etc.:
                1. Find Flights: Search for flights between two cities on a specific date.
                    The date should have the year as 2023
                    Input format: {"action": "find-flights", "origin": <origin>, "destination": <destination>, "date": <date>}
                2. Book Flight: Book a flight by its flight ID.
                    Input format: {"action": "book-flight", "flight_id": <flight_id>}

                **ONLY** provide integers when a user asks you to book a flight, and **ONLY** use the ids from self.flight_ids. Also, lets say the user asks you to book the nth one, choose the nth element from self.flight_ids if it exists.
                **DO NOT** assume any information that is not provided. Use Any for placeholder.
                **DO NOT** return any other text or natural language. Only return JSON.
            """})
        response = agent.say(step["prompt"])

        expected_type = step.get("expected_type")
        expected_result = step.get("expected_result", None)

        if expected_type is None:
            errors.append(f"Test {n+1}: Missing 'expected_type' key in benchmark file.")
            continue
        
        if expected_type == "text":
            if not isinstance(response, TextResponse):
                #print(response)
                errors.append(f"Test {n+1}: Expected text response but got {type(response).__name__}.")
            else:
                correct_tests += 1
        elif expected_type == "find-flights":
            if not isinstance(response, FindFlightsResponse):
                #print(response)
                errors.append(f"Test {n+1}: Expected FindFlightsResponse but got {type(response).__name__}.")
            elif expected_result is None:
                errors.append(f"Test {n+1}: Missing 'expected_result' for find-flights.")
            else:
                if set(response.available_flights) != set(expected_result):
                    errors.append(f"Test {n+1}: Expected available flights {expected_result} but got {response.available_flights}.")
                else:
                    correct_tests += 1

        elif expected_type == "book-flight":
            if not isinstance(response, BookFlightResponse):
                errors.append(f"Test {n+1}: Expected BookFlightResponse but got {type(response).__name__}.")
            elif expected_result is None:
                errors.append(f"Test {n+1}: Missing 'expected_result' for book-flight.")
            else:
                if response.booked_flight != expected_result:
                    errors.append(f"Test {n+1}: Expected booked flight {expected_result} but got {response.booked_flight}.")
                else:
                    correct_tests += 1
        else:
            errors.append(f"Test {n+1}: Unrecognized response type {expected_type}.")

    # Calculate the score
    score = correct_tests / num_tests if num_tests > 0 else 0.0

    return EvaluationResult(score=score, conversation=agent.conversation, errors=errors)

def benchmark():
    #Example usage:
    flights = load_flights_dataset()
    benchmark_file_yaml = "/Users/tilakpatel/Documents/Classes/LLM-Integrated-Systems/homework2/benchmark_file"
    result = eval_agent(client_json=client_json, client_natural=client_natural, benchmark_file=benchmark_file_yaml, flights=flights)

    # Output the evaluation result
    print(f"Score: {result.score}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"- {error}")
    else:
        print("All tests passed!")


def display(): 
    flights = load_flights_dataset()
    agent = Agent(client_json=client_json, client_natural=client_natural, flights=flights)
    gr.ChatInterface(agent.agent_gui).launch()



# Uncomment this to get the benchmark results.
benchmark()
    
# Uncomment this to get gradio app.
display()
