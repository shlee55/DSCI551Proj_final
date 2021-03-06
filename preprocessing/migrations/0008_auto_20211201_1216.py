# Generated by Django 3.2.9 on 2021-12-01 20:16

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0007_mlscore_file_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='predictTable',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('modelType', models.CharField(max_length=100)),
                ('targetVariable', models.CharField(max_length=100)),
                ('file_name', models.CharField(blank=True, max_length=200)),
                ('pub_date', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.AddField(
            model_name='mlscore',
            name='pub_date',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]
