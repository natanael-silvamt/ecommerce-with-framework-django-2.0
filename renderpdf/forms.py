from django import forms
import datetime

class FormDatePdf(forms.Form):
    date = forms.DateField(label='Data', initial=datetime.date.today)
