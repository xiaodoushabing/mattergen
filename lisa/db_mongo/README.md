# MongoDB Setup and Querying Guide

This guide covers setting up MongoDB, connecting to it, and querying data. It also includes common MongoDB query syntax.

## Table of Contents
1. [Setting Up MongoDB](#setting-up-mongodb)
2. [Connecting to MongoDB](#connecting-to-mongodb)
3. [Common MongoDB Syntax](#common-mongodb-syntax)
4. [Querying Results](#querying-results)

---

## Setting Up MongoDB

To set up MongoDB in a Docker container, follow these steps:

1. **Pull the MongoDB Docker image:**
   First, pull the latest MongoDB image from Docker Hub:

   ```bash
   docker pull mongo
   ```

2. **Run MongoDB container**:
    Start a MongoDB container using the following command:

    ```bash
    docker run --name mongo -it -p 27017:27017 mongo
    ```
    This will start MongoDB on port 27017 (default), and the container will be named mongo. If `-itd`, container will be run in detached mode.

3. **Check if MongoDB is running:**
    You can verify that MongoDB is running by listing the active containers:

    ```bash
    docker ps
    ```
    If MongoDB is running, you should see it listed.

## Connecting to MongoDB
From MongoDB Docker Container. To connect to the MongoDB shell inside the container:

1. **Access the MongoDB container:**

    ```bash
    docker exec -it mongo mongosh
    ```
    This opens the MongoDB shell within the container.
    <br>

    If need to connect to MongoDB shell explicitly, use:
    ```bash
    mongosh
    ```

2. **Select the database:** Once inside the MongoDB shell, select the database you want to work with:

    ```bash
    use <your-database-name>
    ```
    For example:
    ```bash
    use mattergen
    ```
3. **Check collections:** To view the collections in the current database:
    ```bash
    show collections
    ```

## Common MongoDB Syntax
#### Show Databases
To list all databases:
```bash
show dbs
```

#### Switch Database
To switch to a specific database:
```bash
use <database-name>
```
*Note: Database will be created if it doesn't already exist.*

#### Insert Document
To insert a document into a collection:

```bash
db.<collection-name>.insertOne({
    "field1": "value1",
    "field2": "value2"
})
```

#### Count Documents
To count the number of documents in a collection:

```bash
db.<collection-name>.countDocuments()
```

#### Aggregation (Group and Count)
You can perform aggregation operations. For example, to group by a field and count the number of documents in each group:

```bash
db.lattices.aggregate([
   { $group: { _id: "$guidance_factor", count: { $sum: 1 } } }
])
```

#### Exiting the MongoDB Shell
To exit the MongoDB shell, simply type:

```bash
exit
```

## Querying Results
[Cheatsheet here :)](https://gist.github.com/codeSTACKr/53fd03c7f75d40d07797b8e4e47d78ec)
#### Basic Query
You can query the documents in a collection using the `find()` method:

```bash
db.<collection-name>.find()
```

For example, to query the lattices collection:

```bash
db.lattices.find()
```

#### Pretty Output
To make the output more readable, use the `pretty()` method:

```bash
db.lattices.find().pretty()
```

#### Query with Conditions
To filter results, pass a query object to `find()`. For example, to find documents with a specific guidance_factor:

```bash
db.lattices.find({ "guidance_factor": 3.0 }).pretty()
```

#### Limiting Results
You can limit the number of results returned by using the `limit()` method. For example, to get only the first 5 results:

```bash
db.lattices.find().limit(5).pretty()
```

#### Sorting Results
To sort the results, use the `sort()` method. For example, to sort by guidance_factor in descending order:

```bash
db.lattices.find().sort({ "guidance_factor": -1 }).pretty()
```

#### Updating Documents
You can update documents with the `updateOne()` or `updateMany()` methods. For example, to update a document:

```bash
db.lattices.updateOne(
   { "guidance_factor": 3.0 },
   { $set: { "magnetic_density": 4.0 } }
)
```

Use `upsert` it you want to also insert if not found
```bash
db.lattices.updateOne(
    { "guidance_factor": 3.0 }, 
    { $set: { "magnetic_density": 4.0 } },
    {
    ### update, if not insert
    upsert: true
    }
)
```
#### Deleting Documents
You can delete documents using the `deleteOne()` or `deleteMany()` methods. For example, to delete a document with a specific guidance_factor:

```bash
db.lattices.deleteOne({ "guidance_factor": 3.0 })
```

#### Greater Than and Less Than
```bash
db.posts.find({ views: { $gt: 2 } })
db.posts.find({ views: { $gte: 7 } })
db.posts.find({ views: { $lt: 7 } })
db.posts.find({ views: { $lte: 7 } })
```

## Troubleshooting
Connection issues: Make sure MongoDB is running and accessible from the container or host.

Permissions: Ensure the MongoDB user has the necessary permissions to access the database or collection.

Firewall: If you are trying to access MongoDB remotely, ensure that the firewall allows incoming traffic on port 27017.

