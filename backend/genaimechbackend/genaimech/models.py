from mongoengine import Document, StringField, EmailField, DateTimeField
from datetime import datetime

class User(Document): 
    name = StringField(max_length=100, required=True)
    phone_number = StringField(max_length=100, required=True)
    email = EmailField(max_length=100, required=True, unique=True)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    reset_token = StringField(max_length=100, default="")
    reset_token_expires = DateTimeField(null=True)
    hashed_password = StringField(max_length=100, required=True)
    salt = StringField(max_length=100, required=True)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super(User, self).save(*args, **kwargs)

    def __str__(self):
        return self.name
