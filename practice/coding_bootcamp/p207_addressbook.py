class AddressBook :
    person_list = []

    def add(self, person):
        self.person_list.append(person)
        
    def show_all(self):
        for person in self.person_list:
            print(person.lastname + " " + person.firstname)
    def search(self,keyword):
        for person in self.person_list:
            if keyword in person.firstname or keyword in person.lastname:
                print(person.lastname, person.firstname)
    
        
    def import_data(self,file):
        import csv
        import datetime
        
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                p=Person()
                p.lastname=row[0]
                p.firstname=row[1]
                p.mail=row[2]

                p.birthday= datetime.datetime.strptime(row[3], '%Y/%m/%d')
                p.tel = row[4]

                self.person_list.append(p)
        
class Person:
    import datetime
    
    firstname=''
    lastname=''
    tel=''
    mail=''
    birthday=datetime.datetime(2000,1,1)

ab = AddressBook()

ab.import_data('person.csv')

'''
p = Person()
p.firstname = '철수'
p.lastname = '김'
p.tel = '010-1234-5678'

ab.add(p)

p2 = Person()
p2.firstname='John'
p2.lastname='Lennon'
p2.tel='010-1234-0098'

print('--동작확인--')
ab.add(p2)
'''
print('주소록에 등록된 사람 수 : ' , str(len(ab.person_list)), '명')
ab.search('철수')
'''
print('--목록표시--')
ab.show_all()

print('--검색--')
ab.search('John')
'''