users = [
{"id" :  0 , 'name" : "Hero" }
{"id" : 1 , 'name" : "Dunn" }
{"id" : 2, 'name" : "Sue"}
{"id" : 3, 'name" : "Chi"}
{"id" : 4, 'name" : "Thor"}
{"id" : 5, 'name" : "Clive"}
{"id" : 6, 'name" : "Hicks"}
{"id" : 7, 'name" : "Devin"}
{"id" : 8, 'name" : "Kate"}
{"id" : 9, 'name" : "Klein"} ]

friendship_paris = [(0,1), (0,2), (1,2), (1,3), (2,3), (3,4), (4,5), (5,6), (5,7), (6,8),(7,8),(8,9)]

frindship = {user['id'] : [] for user in users}

for i, j in frindship_pairs:
  friendships[i].append(j)
  friendships[j].append(i)
  
  
def number_of_friends(user):
  user_id = user["id"]
  friend_ids = friendships[user_id]
  return len(friend_ids)

total_connections = sum(number_of_firneds(user)
                    for user in users)
                    
num_users = len(users)
avg_connections = total_connections / num_users

num_firneds_by_id = [(user["id"], number_of_friends(user))
                    for user in users]

num_friends_by_id.sort(
        key=lambda id_and_firneds : id_and_friends[1],
        reserve=True)
        
def foaf_ids_bad(user):
  return [foaf_id
          for friend_id in friendships[user["id"]]
          for foaf_id in friendships[friend_id]]
print(friendships[0])
print(friendships[1])
print(friendships[2])
                    
from collections import Counter
                    
def friends_of_friends(user) : 
     user_id = user["id"]
     return Counter(
       foaf_id
       for friend_id in friendships[user_id]
       foaf foat_id != user_id
       and fof_id not in frienships[user_id]
     )
print(friends_of_friends(users[3]))

interests = [
  (0, "Haddop"), (0,"Big Data"), (0, "HBase"), (0,"Java"),
  (0, "spark") , (0, "storm", (0, "Cassandra"), 
  (1, "NoSQL"), (1, "MongoDB"), (1,"cassandra"), (1, "HBase"),
  (1, "Postgres"), (1, "Python"), (2,"scikit_learn"), (2,"scipy"),
  (2, "numpy"),(2,"statsmodels"), (2,"pandas), (3,"R"), (3,"python")
  (3,"statistics"), (3, "regression"), (3,"probability")
  (4, "machine learning"), (4,"regression"), (4,"decision trees"),
  (4, "libsvm"),(5,"python"),(5,"R"), (5,"Java"), (5,"C++"),
  (5, "Haskell"), (5,"programming languages"), (6,"statistics")
  (6, "probability"),(6,"mathematics"),(6,"theory"),
  (7, "machine learning"), (7,"scikit_learn"), (7,"Mahout")
  (7, "neural networks"), (8,"neural networks"), (8,"deep learning"),
  (8, "Big Data"), (8, "artificial intelligence"), (9, "Haddop")
  (9, "Java"), (9, "MapReduce"), (9,"Big Data")]
                                   
def data_scientists_who_like(target_interest):
  return [user_id
          for user_id, user_interest in interests
          if user_interest == target_interest]
                                   

# from collections import defaultdict
# user_ids_by_interest = defaultdict(list)

# for user_id, interest in interests : 
#   user_ids_by_interest[interest]append(user_id)
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
  interests_by_user[user_id].append(interest)
                                   
def most_common_interests_with(user):
 return Counter(
   interested_user_id
   for interest in interests_by_id[user"id"]
   for interested_user_id in user_ids_by_interest[interest]
   if interested_user_id != user["id"]
 )
                                   
                                   
