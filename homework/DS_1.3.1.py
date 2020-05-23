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
{"id" : 9, 'name" : "Klein"}

friendship_paris = [(0,1), (0,2), (1,2), (1,3), (2,3), (3,4), (4,5), (5,6), (5,7), (6,8),(7,8),(8,9)

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
        

