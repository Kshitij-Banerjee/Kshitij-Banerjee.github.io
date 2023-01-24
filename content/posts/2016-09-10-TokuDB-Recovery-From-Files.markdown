---
layout: post
title:  "Reverse Engineer data from raw database files. "
description: "How to recover data from raw .tokudb files! Corrupted you tokudb mysql instance ? This post can help you recover the data from just the tokudb files."
author: "Kshitij Banerjee"
avatar: "img/authors/K_icon.png"
img : "toku_db.png"
categories: Database
categories: [Database, Reverse Engineering]
color: 268bd2
---

How to recover data from raw .tokudb files.
=================================

Why?
---------

- Recently my tokudb database went corrupt after a bad shutdown and a lot of data was now lost. After a lot of googling, asking on forums [check here](https://groups.google.com/forum/#!topic/tokudb-user/RrE5MNJFCxw), [here](http://stackoverflow.com/questions/32764692/restore-recover-recreate-tokudb-table-from-missing-status-file) and panicking in general, I finally figured out how to get my data back after some Hard core. Brute force. Raw file Reverse-Engineering.

How?
------------------

1. **Step 1 : Find your raw data files.**
  * The tokufiles have an extension of .tokudb and will be found in your mysql data directory. /var/lib/mysql if you follow the standard conventions.
  * Toku keeps _multiple files per tables_ for data and indices, unlike innodbs combined .ibd files.
        * Status file : example format:  (_database_table_status_table_(some hash)_1_X.tokudb)
        * Main file (*this has the data* - example format : \_database_table_main_table_(som hash)_2_X.tokudb)
        * One file per secondary index of thpe table. (this has the indexes - example format :- \_database_table_key_somethinsomething.tokudb)
  * Another thing to note is that usually the files will have table names in them the _first time_ they are created, but if your ever had an alter creating a temp table on them, the new table doesn't get the name due to a known bug, so not always will you see the table name in the files)
      <img src="http://i.imgur.com/NrCwJkN.png" style="width: 100%"/>
2. **Step 2 : Map the files to the tables**
  * How to know which files correspond to which tables then ?
        - Fortunately, tokudb keeps a map oef the table to file names in the *information_schema.TokuDB_file_map* table!
        - If you have a slave (even broken) with the same tables , you can run this command on it to figure out which files map to which tables.



**Everything below is a fat Step 3: Reverse Engineer the files**
3. Clone the open source repository for toku's fractal trees [here](https://github.com/Tokutek/ft-index)
4. Notice that in the tools directory, you will find a tokuftdump utility. (ft-index/tools/tokuftdump.cc)
  - I suggest going through the utility to understand how the fractal trees are being parsed (This is optional though)
5. Run the utility tool to print out the dump.
  - You should notice that the leafs are printed out as byte streams. These bytestreams contain the complete row of your table in tokudb's internal structure!
6. We want to convert the byte data into meaningful data. So we'll have to first understand the structure of the this bytestream.We need to do some reverse engineering for this.
  - Modify the utility to dump hex streams instead of the byte streams.
  - Use sprintf(hexstr, "%02x", array[i]); [SO link](http://stackoverflow.com/questions/14050452/how-to-convert-byte-array-to-hex-string-in-visual-c)
7. Once you have the hex stream, lets get down to some grammer cracking reverse engineering.
8. Download a tool called 010 Editor from [here](http://www.sweetscape.com/010editor/)
9. Copy one of the rows of the table into the hex window as shown below and use the .bt files to come up with the structure of the row.
  - This step might need you to experiment a bit with your pointer placements and watch the value convertions carefully.  <img src="http://i.imgur.com/E9lDmPU.png" alt="example decode" style="width: 100%;"/>
10. Some things to note.
  - Varchars are padded at the end of the rows.
  - Just before the varchars, you will notice a bunch of numbers that count the size of the ith varchar in your schema.
  - This is because the varchars are only trying to store the actual characters to save space.
  - Once youre happy with the reverse engineering, note this grammer of the data value.
11. Modify the toku print methods to cast the bytestream into your struct as shown below.
![code that uses your struct](http://i.imgur.com/cK5sxNh.png)
12. Run your utility again and voila!, you'll have your data back!


This covers the basics. If you're really stuck and need help. Comment below or reach out to me and I'll try to help, coz I know how much we love our data. ;)
