import datetime
from utils.settings import BUSINESS_EMAIL

# Comprehensive Production Employee Directory: 4 Cities, 3 Areas Each = 12 Employees
EMPLOYEE_DIRECTORY = [
    # --- KARACHI EMPLOYEES ---
    {
        "id": "emp_01",
        "name": "Ahmed Khan",
        "city": "Karachi",
        "areas": ["DHA", "Clifton", "Bahria Town"],
        "phone": "+92 300 1234567",
        "email": BUSINESS_EMAIL,
        "busy_slots": [] 
    },
    {
        "id": "emp_02",
        "name": "Bilal Ahmed",
        "city": "Karachi",
        "areas": ["Gulshan-e-Iqbal", "North Nazimabad", "Gulistan-e-Jauhar"],
        "phone": "+92 321 9876543",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    },
    {
        "id": "emp_03",
        "name": "Zainab Malik",
        "city": "Karachi",
        "areas": ["PECHS", "Tariq Road", "Federal B. Area"],
        "phone": "+92 332 5554321",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    },

    # --- LAHORE EMPLOYEES ---
    {
        "id": "emp_04",
        "name": "Usman Tariq",
        "city": "Lahore",
        "areas": ["DHA", "Gulberg", "Model Town"],
        "phone": "+92 333 4567890",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    },
    {
        "id": "emp_05",
        "name": "Hamza Ali",
        "city": "Lahore",
        "areas": ["Johar Town", "Bahria Town", "Wapda Town"],
        "phone": "+92 345 6789012",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    },
    {
        "id": "emp_06",
        "name": "Ayesha Siddiqui",
        "city": "Lahore",
        "areas": ["Cantt", "Garden Town", "Faisal Town"],
        "phone": "+92 312 3456789",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    },

    # --- ISLAMABAD EMPLOYEES ---
    {
        "id": "emp_07",
        "name": "Farhan Qureshi",
        "city": "Islamabad",
        "areas": ["Blue Area", "F-7", "E-11"],
        "phone": "+92 301 9871234",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    },
    {
        "id": "emp_08",
        "name": "Sadia Noor",
        "city": "Islamabad",
        "areas": ["Bahria Town", "DHA Phase 2", "G-11"],
        "phone": "+92 322 1122334",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    },
    {
        "id": "emp_09",
        "name": "Danyal Khan",
        "city": "Islamabad",
        "areas": ["I-8", "F-10", "PWD Colony"],
        "phone": "+92 334 7788990",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    },

    # --- FAISALABAD EMPLOYEES ---
    {
        "id": "emp_10",
        "name": "Ali Raza",
        "city": "Faisalabad",
        "areas": ["Peoples Colony", "D Ground", "Satyana Road"],
        "phone": "+92 303 4455667",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    },
    {
        "id": "emp_11",
        "name": "Mariam Batool",
        "city": "Faisalabad",
        "areas": ["Madina Town", "Kohinoor City", "Civil Lines"],
        "phone": "+92 315 8899001",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    },
    {
        "id": "emp_12",
        "name": "Saad Javed",
        "city": "Faisalabad",
        "areas": ["Gulberg", "Samundri Road", "Canal Road"],
        "phone": "+92 323 3344556",
        "email": BUSINESS_EMAIL,
        "busy_slots": []
    }
]

class EmployeeAssignmentManager:
    def __init__(self):
        self.employees = EMPLOYEE_DIRECTORY

    def assign_best_employee(self, city: str, target_area: str, meeting_time_str: str, client_text: str = "") -> dict:
        """
        Smart matching algorithm: 
        1. Checks both target_area and client_text for area keywords.
        2. Matches City + Area.
        3. Falls back to City only or default.
        """
        best_match = None
        combined_search_text = f"{target_area} {client_text}".lower()

        # Step 1: Match City + Area (checking both target_area and client input text)
        if city:
            for emp in self.employees:
                if emp["city"].lower() == city.lower():
                    # Check if any of the employee's areas are mentioned in the text or target_area
                    if any(area.lower() in combined_search_text for area in emp["areas"]):
                        if meeting_time_str not in emp["busy_slots"]:
                            best_match = emp
                            break

        # Step 2: Match City only if no area-specific agent found
        if not best_match and city:
            for emp in self.employees:
                if emp["city"].lower() == city.lower():
                    if meeting_time_str not in emp["busy_slots"]:
                        best_match = emp
                        break

        # Step 3: Absolute Fallback to first employee
        if not best_match and self.employees:
            best_match = self.employees[0]

        # Register the time slot as busy once assigned
        if best_match and meeting_time_str not in best_match["busy_slots"]:
            best_match["busy_slots"].append(meeting_time_str)

        return best_match
