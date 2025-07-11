import sqlite3

# connect to sqlite
connection=sqlite3.connect('student.db')

#create a cursor object to insert record,create table
cursor=connection.cursor()

#create table
table_info="""
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_info)

cursor.execute("Insert into student values('krish','data science','A',90)")
cursor.execute("Insert into student values('alriyan','data science','B',100)")
cursor.execute("Insert into student values('khushi','data science','A',86)")
cursor.execute("Insert into student values('jacob','devops','A',50)")
cursor.execute("Insert into student values('fardeen','devops','A',70)")

print('The inserted records are')
data=cursor.execute("select * from student")
for row in data:
    print(row)


connection.commit()
connection.close()

