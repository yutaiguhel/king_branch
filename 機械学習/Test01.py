class Employee:

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)


class Programmer(Employee):
    raise_amt = 1.10

    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay)
        self.prog_lang = prog_lang


class Manager(Employee):
    raise_amt = 1.50

    def __init__(self, first, last, pay, employees=None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            print('雇っているエンジニア：', emp.fullname(), 'さん')
            print('対応言語：', emp.prog_lang)

e=Employee("Koga", "Yuta", 100000)
dev_1 = Programmer('Tanaka', 'Tarou', 50000, 'Python')
dev_2 = Programmer('Oshima', 'Takayuki', 60000, 'PHP')
mgr_1 = Manager('Adam', 'Jozee', 60000)

print(mgr_1.fullname())
print(mgr_1.email)
print('基本年収（$）：', mgr_1.pay)
mgr_1.apply_raise()
print('年間の役職手当（$）：', mgr_1.pay)

mgr_1.add_emp(dev_1)
mgr_1.print_emps()