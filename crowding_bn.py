from pgmpy.models import DiscreteBayesianNetwork

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def build_bn():
    model = DiscreteBayesianNetwork([
        ("Weather", "Demand"),
        ("Time", "Demand"),
        ("DayType", "Demand"),
        ("Demand", "Crowding"),
        ("Service", "Crowding"),
        ("Mode", "Crowding"),
    ])

    # ---- Priors ----
    cpd_weather = TabularCPD("Weather", 3, [[0.6], [0.3], [0.1]],
                            state_names={"Weather": ["Clear", "Rainy", "Thunderstorms"]})

    cpd_time = TabularCPD("Time", 3, [[0.35], [0.30], [0.35]],
                          state_names={"Time": ["Morning", "Afternoon", "Evening"]})

    cpd_day = TabularCPD("DayType", 2, [[0.7], [0.3]],
                         state_names={"DayType": ["Weekday", "Weekend"]})

    cpd_service = TabularCPD("Service", 3, [[0.7], [0.2], [0.1]],
                             state_names={"Service": ["Normal", "Reduced", "Disrupted"]})

    # For experiments, you can set 0.5/0.5
    cpd_mode = TabularCPD("Mode", 2, [[0.5], [0.5]],
                          state_names={"Mode": ["Today", "Future"]})

    # ---- Demand CPT: Demand | Weather, Time, DayType ----
    # Demand states: Low, Medium, High
    # Order of parents: Weather, Time, DayType
    # You MUST provide 3 rows (Low/Med/High) and 3*3*2 = 18 columns.
    # Tip: start with something reasonable, then adjust.

    demand_cols = 18
    # Start with a simple pattern: base Medium, push High for (Weekend+Evening) and (Rain/Storm)
    # (These numbers are example assumptions; tune them.)
    low =    []
    med =    []
    high =   []

    weather_states = ["Clear", "Rainy", "Thunderstorms"]
    time_states = ["Morning", "Afternoon", "Evening"]
    day_states = ["Weekday", "Weekend"]

    for w in weather_states:
        for t in time_states:
            for d in day_states:
                # base
                p_low, p_med, p_high = 0.25, 0.55, 0.20

                # weekend + evening -> more high
                if d == "Weekend" and t == "Evening":
                    p_low, p_med, p_high = 0.15, 0.45, 0.40

                # rainy -> slightly more high
                if w == "Rainy":
                    p_low -= 0.05
                    p_high += 0.05

                # thunderstorms -> more high
                if w == "Thunderstorms":
                    p_low -= 0.10
                    p_high += 0.10

                # keep within [0,1] and sum=1 (quick clamp)
                p_low = max(0.05, min(0.80, p_low))
                p_high = max(0.05, min(0.80, p_high))
                p_med = 1.0 - p_low - p_high

                low.append(p_low)
                med.append(p_med)
                high.append(p_high)

    cpd_demand = TabularCPD(
        "Demand", 3,
        [low, med, high],
        evidence=["Weather", "Time", "DayType"],
        evidence_card=[3, 3, 2],
        state_names={
            "Demand": ["Low", "Medium", "High"],
            "Weather": weather_states,
            "Time": time_states,
            "DayType": day_states
        }
    )

    # ---- Crowding CPT: Crowding | Demand, Service, Mode ----
    # Crowding states: Low, Medium, High
    # Parents: Demand (3), Service (3), Mode (2) => 18 columns
    crow_low, crow_med, crow_high = [], [], []

    demand_states = ["Low", "Medium", "High"]
    service_states = ["Normal", "Reduced", "Disrupted"]
    mode_states = ["Today", "Future"]

    for dem in demand_states:
        for serv in service_states:
            for mode in mode_states:
                # base: tie crowding to demand
                if dem == "Low":
                    pL, pM, pH = 0.70, 0.25, 0.05
                elif dem == "Medium":
                    pL, pM, pH = 0.25, 0.55, 0.20
                else:  # High demand
                    pL, pM, pH = 0.10, 0.35, 0.55

                # service problems increase crowding
                if serv == "Reduced":
                    pL -= 0.05; pH += 0.05
                if serv == "Disrupted":
                    pL -= 0.10; pH += 0.10

                # Future mode assumption (pick ONE story and keep consistent)
                # Example assumption: Future reduces crowding slightly (better connectivity/spreads load)
                if mode == "Future":
                    pL += 0.05; pH -= 0.05

                # clamp + renormalize
                pL = max(0.05, min(0.90, pL))
                pH = max(0.05, min(0.90, pH))
                pM = 1.0 - pL - pH

                crow_low.append(pL)
                crow_med.append(pM)
                crow_high.append(pH)

    cpd_crowd = TabularCPD(
        "Crowding", 3,
        [crow_low, crow_med, crow_high],
        evidence=["Demand", "Service", "Mode"],
        evidence_card=[3, 3, 2],
        state_names={
            "Crowding": ["Low", "Medium", "High"],
            "Demand": demand_states,
            "Service": service_states,
            "Mode": mode_states
        }
    )

    model.add_cpds(cpd_weather, cpd_time, cpd_day, cpd_service, cpd_mode, cpd_demand, cpd_crowd)
    model.check_model()

    return model

def query_crowding(evidence: dict):
    model = build_bn()
    infer = VariableElimination(model)
    q = infer.query(["Crowding"], evidence=evidence)
    return q

if __name__ == "__main__":
    evidence = {
        "Weather": "Rainy",
        "Time": "Evening",
        "DayType": "Weekend",
        "Service": "Reduced",
        "Mode": "Today",
    }
    print(query_crowding(evidence))
