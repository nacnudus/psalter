import os
import json
import csv

psalms = []
fieldnames = [
    "psalm_number",
    "section_number",
    "section_title",
    "verse_number",
    "part1",
    "part2",
]

with open("psalms.csv", "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    with open("psalms.json") as json_file:
        psalms = json.load(json_file)

        for psalm in psalms:
            psalm_number = psalm["number"]

            for section in psalm["sections"]:
                section_number = section["number"]
                section_title = section["title"]

                for verse in section["verses"]:
                    verse_number = verse["number"]
                    part1 = verse["part1"]
                    part2 = verse["part2"]

                    writer.writerow(
                        {
                            "psalm_number": psalm_number,
                            "section_number": section_number,
                            "section_title": section_title,
                            "verse_number": verse_number,
                            "part1": part1,
                            "part2": part2,
                        }
                    )
