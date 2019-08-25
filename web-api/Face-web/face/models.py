from django.db import models
import os
import pandas as pd

# Create your models here.

class StarInfo(models.Model):
    star_id = models.PositiveIntegerField(primary_key=True)
    name = models.CharField(max_length=200)
    e_name = models.CharField(max_length=200)
    star_type = models.CharField(max_length=50)
    star_other = models.CharField(max_length=50)
    area = models.CharField(max_length=50)
    desc = models.CharField(max_length=50)
    reels = models.CharField(max_length=1000)
    sex = models.IntegerField(default=0)

    def __str__(self):
        return str(self.star_id)
