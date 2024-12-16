import json


# The purpose of this class is to generate a json file containing, for each person (identified with an ID),
# information regarding gender, whether they are wearing a hat, whether they are wearing a bag and the trajectory.
class OutputWriter:
    def __init__(self):
        # Initialize a list to store the information about people
        self.people = []

    def add_person(self, person_id, gender, hat, bag, trajectory):
        """
        Add a person and their related information

        :param person_id: int, person ID
        :param gender: str, "male" or "female"
        :param hat: bool, indicates whether the person is wearing a hat or not
        :param bag: bool, indicates whether the person is wearing a bag or not
        :param trajectory: list, a list of virtual line IDs, namely the ordered sequence of lines crossed by the person
        """
        person = {
            "id": person_id,
            "gender": gender,
            "hat": hat,
            "bag": bag,
            "trajectory": trajectory
        }
        self.people.append(person)

    def write_output(self, filename):
        """
        Write the stored information in a JSON file.

        :param filename: str, name of the output file
        """
        output = {"people": self.people}
        with open(filename, 'w') as file:
            json.dump(output, file, indent=4)


# Test, da eliminare prima della consegna.
# if __name__ == "__main__":
#     writer = OutputWriter()
#     writer.add_person(1, "male", True, False, [1, 2, 3, 4])
#     writer.add_person(2, "female", False, True, [4, 1, 3, 1, 2])
#     writer.write_output("output.json")