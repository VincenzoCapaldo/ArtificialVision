import json


# The purpose of this class is to generate a json file containing, for each person (identified with an ID),
# information regarding gender, whether they are wearing a hat, whether they are wearing a bag and the trajectory.
class OutputWriter:
    def __init__(self):
        # Initialize a list to store the information about people
        self.people = []

    def add_person(self, person_id, gender, bag, hat, trajectory):
        """
        Add a new person if the person_id is not already present and their related information

        :param person_id: int, person ID
        :param gender: str, "male" or "female"
        :param hat: bool, indicates whether the person is wearing a hat or not
        :param bag: bool, indicates whether the person is wearing a bag or not
        :param trajectory: list, a list of virtual line IDs, namely the ordered sequence of lines crossed by the person
        """
        for person in self.people:
            if person["id"] == person_id:
                print(f"Person with ID {person_id} already exists.")
                return

        if gender == 0:
            gender = "male"
        else:
            gender = "female"

        if bag == 0:
            bag = False
        else:
            bag = True

        if hat == 0:
            hat = False
        else:
            hat = True

        trajectory = map(str, trajectory) if trajectory is not None else '[]'

        person = {
            "id": person_id,
            "gender": gender,
            "hat": hat,
            "bag": bag,
            "trajectory": trajectory
        }
        self.people.append(person)

    def write_output(self, filename="./result/result.txt"):
        """
        Write the stored information in a JSON file.

        :param filename: str, name of the result file
        """
        output = {"people": self.people}
        with open(filename, 'w') as file:
            json.dump(output, file, indent=4)
