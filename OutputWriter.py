import json


# The purpose of this class is to generate a json file containing, for each person (identified with an ID),
# information regarding gender, whether they are wearing a hat, whether they are wearing a bag and the trajectory.
class OutputWriter:
    def __init__(self):
        # Initialize a list to store the information about people
        self.people = []

    def add_person(self, person_id, gender="", hat="", bag="", trajectory=""):
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
                return
        person = {
                    "id": person_id,
                    "gender": gender,
                    "hat": hat,
                    "bag": bag,
                    "trajectory": trajectory
                }
        self.people.append(person)

    def set_trajectory(self, person_id, trajectory):
        """
        Update the trajectory of a person identified by person_id.

        :param person_id: int, person ID
        :param trajectory: list, a list of virtual line IDs
        """
        for person in self.people:
            if person["id"] == person_id:
                person["trajectory"] = trajectory
                return
        raise ValueError(f"Person with ID {person_id} not found.")

    def set_gender(self, person_id, gender):
        """
        Update the gender of a person identified by person_id.

        :param person_id: int, person ID
        :param gender: str, "male" or "female"
        """
        for person in self.people:
            if person["id"] == person_id:
                person["gender"] = gender
                return
        raise ValueError(f"Person with ID {person_id} not found.")

    def set_hat(self, person_id, hat):
        """
        Update the hat status of a person identified by person_id.

        :param person_id: int, person ID
        :param hat: bool, indicates whether the person is wearing a hat or not
        """
        for person in self.people:
            if person["id"] == person_id:
                person["hat"] = hat
                return
        raise ValueError(f"Person with ID {person_id} not found.")

    def set_bag(self, person_id, bag):
        """
        Update the bag status of a person identified by person_id.

        :param person_id: int, person ID
        :param bag: bool, indicates whether the person is wearing a bag or not
        """
        for person in self.people:
            if person["id"] == person_id:
                person["bag"] = bag
                return
        raise ValueError(f"Person with ID {person_id} not found.")

    def write_output(self, filename="./output/output.json"):
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