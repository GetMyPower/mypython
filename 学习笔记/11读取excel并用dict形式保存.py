# 读取id、姓名、成绩
# aa.xlsx:
# id||name||score
# 1||mike||88.5
# 2||amy||60.8
# 3||bob||79.6
import xlrd


# region (1)将excel内容存于student类型的list
class student():
    def __init__(self):
        self.id = 0
        self.name = 0
        self.age = 0


def read_student(filename):
    workbook = xlrd.open_workbook(filename)
    sheet = workbook.sheet_by_index(0)  # 最前边的sheet

    Students = []
    for i in range(1, sheet.nrows, 1):  # 从第2行开始读，步长是1
        row = sheet.row_values(i)
        a_student = student()
        a_student.id = int(row[0])
        a_student.name = row[1]
        a_student.age = row[2]
        Students.append(a_student)
    return Students


Students_lst = read_student("aa.xlsx")


# endregion


# region (2)将excel内容存于dict形式的list
def read_dict(filename):
    workbook = xlrd.open_workbook(filename)
    sheet = workbook.sheet_by_index(0)  # 最前边的sheet

    Students = []
    for i in range(1, sheet.nrows, 1):  # 从第2行开始读，步长是1
        row = sheet.row_values(i)
        a_stu = {}
        a_stu['id'] = int(row[0])
        a_stu['name'] = row[1]
        a_stu['score'] = float(row[2])
        Students.append(a_stu)
    return Students


Students_dict = read_dict("aa.xlsx")
# endregion

print()
