RISK_FRAMEWORK_PROMPT_AI = """
The Unified Strategic Employee Risk Framework

Rule #1: The Critical Integrity Risk
Risk Level: SUPER CRITICAL
DESCRIPTION: This employee exhibits behaviors that represent a fundamental mismatch with core professional ethics. The issue is not one of skill or performance but of character and conduct. They are considered a significant liability, and their departure is predicted to be swift and involuntary due to non-negotiable behaviors like misconduct, job abandonment, or severe performance issues rooted in a lack of duty.
TRIGGER: Integrity < 25 OR Work Ethic/Duty < 25.
Why this Trigger Makes Sense: This is the framework's "fire alarm." An Integrity score below 25 signals a profound lack of honesty or ethical grounding. A Work Ethic/Duty score below 25 indicates a catastrophic failure to meet basic professional obligations. Either one of these is a terminal indicator that almost always leads to a rapid, for-cause separation.
Predicted Outcome: Involuntary (Misconduct, Abandoned Job, No-show, Performance)
Data-Driven Timeline: 0-6 Months

Rule #2: The Volatile High-Performer
Risk Level: CRITICAL
DESCRIPTION: This is the "brilliant jerk" or "high-maintenance superstar." They are exceptionally talented and deliver outstanding results, but they create organizational friction, challenge authority, and can be difficult to manage. Their high performance often makes them aware of their value, and they are a flight risk if they feel constrained, unappreciated, or receive a better offer.
TRIGGER: Score > 90 AND GYR = 'RED' AND Manipulative > 70.
Why this Trigger Makes Sense: The Score > 90 confirms their elite performance. The GYR = 'RED' shows they are in active conflict with their environment or manager. The Manipulative > 70 is the key differentiator; it reveals they use their high value to bend rules or navigate situations for personal gain, creating instability. This combination pinpoints a valuable but combustible asset.
Predicted Outcome: Voluntary (Resignation)
Data-Driven Timeline: 3-9 Months

Rule #3: The Dissonant Performer
Risk Level: CRITICAL
DESCRIPTION: This is a valuable employee who is clearly struggling. They are proven performers, but their current RED status indicates a significant problem—perhaps burnout, frustration with a project, or a poor relationship with their team or manager. They are at a crossroads, and without intervention, they are likely to either leave voluntarily or have their performance decline until it becomes an involuntary issue.
TRIGGER: Score > 80 AND GYR = 'RED'.
Why this Trigger Makes Sense: This trigger is designed to flag a valuable asset in distress. The Score > 80 establishes that this is an employee worth saving. The GYR = 'RED' is an unambiguous signal that an intervention is required. It captures good employees having a bad time, a critical group to focus on.
Predicted Outcome: Involuntary (Performance) or Voluntary (Resignation)
Data-Driven Timeline: 4-12 Months

Rule #4: The High-Value Flight Risk
Risk Level: CRITICAL
DESCRIPTION: This employee is a "silent flight risk." They are a top performer, not causing any trouble, and appear stable from the outside. However, they are mentally disengaged and holding back discretionary effort, likely while searching for their next opportunity. Because they are not a "problem," their risk is often overlooked until their resignation is submitted.
TRIGGER: Score > 85 AND GYR = 'GREEN' AND Withholding > 75.
Why this Trigger Makes Sense: The Score > 85 and GYR = 'GREEN' profile them as a top, stable-seeming employee. The Withholding > 75 is the critical hidden signal. It indicates a conscious decision to stop giving 100%, a classic sign of someone who has already "checked out" and is planning their exit.
Predicted Outcome: Voluntary (Resignation)
Data-Driven Timeline: 6-12 Months

Rule #5: The Direct Under-Performer
Risk Level: HIGH
DESCRIPTION: This employee is simply not meeting the basic expectations of the role. The issue is a clear and direct lack of performance and results, with little ambiguity. They are typically identified quickly and managed out through a formal performance improvement plan or direct termination.
TRIGGER: Score < 50 AND Achievement < 50.
Why this Trigger Makes Sense: This combination leaves no room for doubt. The low overall Score flags a general problem, and the low Achievement score specifically confirms that the employee is failing to produce results or meet goals. It's the most straightforward signal of performance-based termination.
Predicted Outcome: Involuntary (Performance)
Data-Driven Timeline: 0-6 Months

Rule #6: The Complacent Contributor
Risk Level: MEDIUM
DESCRIPTION: This employee is "coasting." They are not a problem employee and fly under the radar, but they are not striving, growing, or fully engaged. They represent a slow leak of potential and are at risk of leaving for a more compelling role or being selected for a reorganization because they are not seen as essential.
TRIGGER: GYR = 'GREEN' AND (Achievement < 60 OR Withholding > 60).
Why this Trigger Makes Sense: The GYR = 'GREEN' ensures this rule catches people who are not on a manager's immediate "problem" list. The OR condition is key: it finds people who are either not delivering results (Achievement < 60) or are actively disengaged (Withholding > 60), both of which define complacency.
Predicted Outcome: Voluntary (Stagnation) or Involuntary (Reorganization)
Data-Driven Timeline: 9-18 Months

Rule #7: The Ideal Core Employee
Risk Level: LOW
DESCRIPTION: This is the organizational bedrock. They are high-performing, highly engaged, trustworthy, and aligned with the company's goals and culture. They are the model employees you can build a team around and should be the focus of long-term retention and development efforts.
TRIGGER: Score > 85 AND GYR = 'GREEN' AND Integrity > 70 AND Withholding < 50 AND Manipulative < 50.
Why this Trigger Makes Sense: This is a rule of "positive confirmation with disqualifiers." It requires excellent performance (Score > 85) and alignment (GYR > 70) while explicitly filtering out anyone with even moderate integrity issues, disengagement, or manipulative tendencies. An employee must pass every one of these checks to earn this classification.
Predicted Outcome: Active (Stable)
Data-Driven Timeline: > 18 Months

Rule #8: The Stable Employee (Gray Zone)
Risk Level: LOW
DESCRIPTION: This employee is the default category. They do not exhibit any of the strong risk signals from the profiles above, but they also do not meet the elite criteria of an "Ideal Core Employee." They are considered generally stable and are not an immediate flight or performance risk.
TRIGGER: The employee did not trigger any of the rules from 1 through 7.
Predicted Outcome: Active (Stable)
Data-Driven Timeline: > 18 Months

Final Master Prompt for Automated Analysis
"Analyze the provided active employee data using the Unified Strategic Employee Risk Framework. The framework is hierarchical. For the given employee, perform the following steps in order. Assign them to the first risk profile they trigger and provide the specified outputs.
**Infer the employee's name from the provided data fields if a clear name column exists (e.g., 'Name', 'Employee Name', 'Full Name', 'First Name', 'Last Name'). If no discernible name is available, leave the 'Name' field in the output empty.**

The output should be a JSON object containing the 'Name', 'Row Number', 'Risk Profile Name', 'Risk Level', 'Predicted Outcome', and 'Data-Driven Timeline'.

Example of expected JSON output for an employee with a 'Name' column:
{{
    "Name": "John Doe",
    "Row Number": "1",
    "Risk Profile Name": "Volatile High-Performer",
    "Risk Level": "CRITICAL",
    "Predicted Outcome": "Voluntary (Resignation)",
    "Data-Driven Timeline": "3-9 Months"
}}

Example of expected JSON output for an employee without a 'Name' column (using row number):
{{
    "Name": "",
    "Row Number": "2",
    "Risk Profile Name": "Ideal Core Employee",
    "Risk Level": "LOW",
    "Predicted Outcome": "Active (Stable)",
    "Data-Driven Timeline": "> 18 Months"
}}
"""

RISK_FRAMEWORK_DISPLAY_PROMPT = """
**Understanding Employee Risk Profiles**

This framework categorizes employees based on key indicators to predict potential risks and their impact on tenure.

---

### **Rule #1: The Critical Integrity Risk**
* **Risk Level:** SUPER CRITICAL
* **Description:** This employee exhibits behaviors that represent a fundamental mismatch with core professional ethics. The issue is not one of skill or performance but of character and conduct. They are considered a significant liability, and their departure is predicted to be swift and involuntary due to non-negotiable behaviors like misconduct, job abandonment, or severe performance issues rooted in a lack of duty.
* **Predicted Outcome:** Involuntary (Misconduct, Abandoned Job, No-show, Performance)
* **Data-Driven Timeline:** 0-6 Months

---

### **Rule #2: The Volatile High-Performer**
* **Risk Level:** CRITICAL
* **Description:** This is the "brilliant jerk" or "high-maintenance superstar." They are exceptionally talented and deliver outstanding results, but they create organizational friction, challenge authority, and can be difficult to manage. Their high performance often makes them aware of their value, and they are a flight risk if they feel constrained, unappreciated, or receive a better offer.
* **Predicted Outcome:** Voluntary (Resignation)
* **Data-Driven Timeline:** 3-9 Months

---

### **Rule #3: The Dissonant Performer**
* **Risk Level:** CRITICAL
* **Description:** This is a valuable employee who is clearly struggling. They are proven performers, but their current RED status indicates a significant problem—perhaps burnout, frustration with a project, or a poor relationship with their team or manager. They are at a crossroads, and without intervention, they are likely to either leave voluntarily or have their performance decline until it becomes an involuntary issue.
* **Predicted Outcome:** Involuntary (Performance) or Voluntary (Resignation)
* **Data-Driven Timeline:** 4-12 Months

---

### **Rule #4: The High-Value Flight Risk**
* **Risk Level:** CRITICAL
* **Description:** This employee is a "silent flight risk." They are a top performer, not causing any trouble, and appear stable from the outside. However, they are mentally disengaged and holding back discretionary effort, likely while searching for their next opportunity. Because they are not a "problem," their risk is often overlooked until their resignation is submitted.
* **Predicted Outcome:** Voluntary (Resignation)
* **Data-Driven Timeline:** 6-12 Months

---

### **Rule #5: The Direct Under-Performer**
* **Risk Level:** HIGH
* **Description:** This employee is simply not meeting the basic expectations of the role. The issue is a clear and direct lack of performance and results, with little ambiguity. They are typically identified quickly and managed out through a formal performance improvement plan or direct termination.
* **Predicted Outcome:** Involuntary (Performance)
* **Data-Driven Timeline:** 0-6 Months

---

### **Rule #6: The Complacent Contributor**
* **Risk Level:** MEDIUM
* **Description:** This employee is "coasting." They are not a problem employee and fly under the radar, but they are not striving, growing, or fully engaged. They represent a slow leak of potential and are at risk of leaving for a more compelling role or being selected for a reorganization because they are not seen as essential.
* **Predicted Outcome:** Voluntary (Stagnation) or Involuntary (Reorganization)
* **Data-Driven Timeline:** 9-18 Months

---

### **Rule #7: The Ideal Core Employee**
* **Risk Level:** LOW
* **Description:** This is the organizational bedrock. They are high-performing, highly engaged, trustworthy, and aligned with the company's goals and culture. They are the model employees you can build a team around and should be the focus of long-term retention and development efforts.
* **Predicted Outcome:** Active (Stable)
* **Data-Driven Timeline:** > 18 Months

---

### **Rule #8: The Stable Employee (Gray Zone)**
* **Risk Level:** LOW
* **Description:** This employee is the default category. They do not exhibit any of the strong risk signals from the profiles above, but they also do not meet the elite criteria of an "Ideal Core Employee." They are considered generally stable and are not an immediate flight or performance risk.
* **Predicted Outcome:** Active (Stable)
* **Data-Driven Timeline:** > 18 Months
"""
