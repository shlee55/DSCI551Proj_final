# Generated by Django 3.2.9 on 2021-12-01 05:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0005_metadata_file_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='metadataforreset',
            name='file_name',
            field=models.CharField(blank=True, max_length=200),
        ),
    ]
