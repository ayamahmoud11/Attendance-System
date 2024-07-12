import mysql.connector

# Function to retrieve student ID from the database based on the student name
def get_student_info(id):
    # Establish a connection to the MySQL server
    cnx = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Administrator1',
        database='attendance_system'
    )

    # Create a cursor object to execute SQL queries
    cursor = cnx.cursor()

    # Execute the SQL query to select student ID by name
    query = f"SELECT first_name,last_name,academic_year,department FROM student WHERE id = '{id}'"
    cursor.execute(query)

    # Fetch the first row (assuming there's only one matching student)
    student_data  = cursor.fetchone()

    # Close the cursor and connection
    cursor.close()
    cnx.close()
    print(student_data)
    return student_data 