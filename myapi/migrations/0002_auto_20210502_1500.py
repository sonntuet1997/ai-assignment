# Generated by Django 3.2 on 2021-05-02 15:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapi', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='sentence',
            name='result',
            field=models.CharField(default=2, max_length=10),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='sentence',
            name='name',
            field=models.CharField(max_length=200),
        ),
        migrations.AlterField(
            model_name='sentence',
            name='type',
            field=models.CharField(max_length=200),
        ),
    ]
