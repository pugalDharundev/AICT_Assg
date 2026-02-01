from crowding_bn import query_crowding

def run_case(name, evidence):
    dist = query_crowding(evidence)   # dist is already the factor
    states = dist.state_names["Crowding"]
    probs = dist.values


    print(f"\n=== {name} ===")
    print("Evidence:", evidence)
    for s, p in zip(states, probs):
        print(f"  P(Crowding={s}) = {p:.3f}")

def main():
    # Scenario 1: compare Today vs Future (same evidence, only Mode changes)
    base1 = {"Weather":"Rainy","Time":"Evening","DayType":"Weekend","Service":"Reduced"}
    run_case("S1 - Today",  {**base1, "Mode":"Today"})
    run_case("S1 - Future", {**base1, "Mode":"Future"})

    # Scenario 2: compare Today vs Future
    base2 = {"Weather":"Clear","Time":"Morning","DayType":"Weekday","Service":"Normal"}
    run_case("S2 - Today",  {**base2, "Mode":"Today"})
    run_case("S2 - Future", {**base2, "Mode":"Future"})

    # Scenario 3: compare Today vs Future
    base3 = {"Weather":"Thunderstorms","Time":"Afternoon","DayType":"Weekend","Service":"Disrupted"}
    run_case("S3 - Today",  {**base3, "Mode":"Today"})
    run_case("S3 - Future", {**base3, "Mode":"Future"})

    # Extra scenario 4 (optional, makes it 7 scenarios total)
    base4 = {"Weather":"Rainy","Time":"Morning","DayType":"Weekday","Service":"Normal"}
    run_case("S4 - Today",  {**base4, "Mode":"Today"})

if __name__ == "__main__":
    main()
