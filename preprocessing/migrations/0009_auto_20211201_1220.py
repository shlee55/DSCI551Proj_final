# Generated by Django 3.2.9 on 2021-12-01 20:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0008_auto_20211201_1216'),
    ]

    operations = [
        migrations.AddField(
            model_name='predicttable',
            name='indepVariable',
            field=models.CharField(blank=True, max_length=100),
        ),
        migrations.AlterField(
            model_name='predicttable',
            name='modelType',
            field=models.CharField(blank=True, max_length=100),
        ),
        migrations.AlterField(
            model_name='predicttable',
            name='targetVariable',
            field=models.CharField(blank=True, max_length=100),
        ),
    ]
