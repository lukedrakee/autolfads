#!/usr/bin/env python3
"""
MongoDB Database Connection Utilities for PBT
Modernized for Python 3.9+ and modern PyMongo
"""

import pymongo
import time


class DatabaseConnection:
    """
    Handles MongoDB database connections and operations for PBT.
    """
    def __init__(self, project_name, host, port, user, password):
        """
        Initialize database connection.

        Args:
            project_name: Name of the MongoDB database
            host: MongoDB server hostname
            port: MongoDB server port
            user: MongoDB username
            password: MongoDB password
        """
        # Modern pymongo connection string
        if user and password:
            connection_string = f'mongodb://{user}:{password}@{host}:{port}/admin'
        else:
            connection_string = f'mongodb://{host}:{port}'

        client = pymongo.MongoClient(connection_string)
        self.db = client[project_name]

    def read_many(self, collection_name, _id, fields):
        """
        Read multiple fields from a document.

        Args:
            collection_name: Name of the collection.
            _id: _id of the document to be read.
            fields: List of fields to be read.

        Returns:
            Dictionary of field values or raises ValueError.
        """
        aggregate_fields = {f: "$" + f for f in fields}
        collection = self.db[collection_name]
        cursor = collection.aggregate([
            {"$match": {"_id": _id}},
            {"$group": {"_id": aggregate_fields}}
        ])
        items = list(cursor)
        if len(items) < 1:
            raise ValueError(f"No document with the given id {_id} was found.")
        elif len(items) > 1:
            raise ValueError(f"More than one document with the given id {_id} was found.")
        else:
            return items[0]['_id']

    def read_one(self, collection_name, _id, field):
        """
        Read a single field from a document.

        Args:
            collection_name: Name of the collection.
            _id: _id of the document to be read.
            field: Field to be read.

        Returns:
            Value of the field or raises ValueError.
        """
        collection = self.db[collection_name]
        cursor = collection.aggregate([
            {"$match": {"_id": _id}},
            {"$group": {"_id": "$" + field}}
        ])
        items = list(cursor)
        if len(items) < 1:
            raise ValueError(f"No document with the given id {_id} was found.")
        elif len(items) > 1:
            raise ValueError(f"More than one document with the given id {_id} was found.")
        else:
            return items[0]['_id']

    def write_many(self, collection_name, _id, fields, values):
        """
        Write multiple fields to a document (upsert).

        Args:
            collection_name: Name of the collection.
            _id: _id of the document to be written to.
            fields: List of fields to be written to.
            values: List of values to be written to fields.

        Returns:
            True for success, raises AssertionError for failure.
        """
        collection = self.db[collection_name]
        update_vals = dict(zip(fields, values))

        collection.update_one(
            {"_id": _id},
            {"$set": update_vals},
            upsert=True
        )

        # Verify write
        item = self.read_many(collection_name, _id, fields)
        for fld in fields:
            assert item[fld] == update_vals[fld], \
                "Mismatch between what was written to, and read from the MongoDB. Write failed."

    def write_one(self, collection_name, _id, field, value):
        """
        Write a single field to a document (upsert).

        Args:
            collection_name: Name of the collection.
            _id: _id of the document to be written to.
            field: Field to be written to.
            value: Value to be written.

        Returns:
            True for success, raises AssertionError for failure.
        """
        collection = self.db[collection_name]

        for _ in range(5):
            # re-try writing with confirmation for 5 times
            collection.update_one(
                {"_id": _id},
                {"$set": {field: value}},
                upsert=True
            )

            item = self.read_one(collection_name, _id, field)

            if item == value:
                return True
            else:
                # re-try every 0.5 second
                time.sleep(0.5)

        raise AssertionError("MongoDB write failed!")

    def read_all(self, collection_name, field):
        """
        Read all distinct values of a field from a collection.

        Args:
            collection_name: Name of the collection.
            field: Field to get distinct values from.

        Returns:
            List of distinct values.
        """
        return self.db[collection_name].distinct(field)
