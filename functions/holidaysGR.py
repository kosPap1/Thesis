import holidays

#finding Greek holidays

def holiday(year):
    holiday = []
    for date in holidays.Greece(years=year):
        holiday.append(str(date))
    return holiday


x=holiday(2015)
print(x)













