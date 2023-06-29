import matplotlib.pyplot as plt
import random
import networkx
import math
import distributions
import numpy as np

class agent:
    def __init__(self,Status,Number,Attitude):
        self.status = Status
        self.number = Number
        self.attitude = Attitude
    status = None
    number = None
    attitude = 0
    day_became_innoculated = None

number_runs = 3
iterations = 500
make_network_image = True
make_attitude_spread = True
density = 0.1
Use_Scale_Free_And_Centers = True
Attitude_Centers = [-1,0,1]
inoculation_selection = 3 #0 is random, 1 is left, 2 is center, 3 is right
#to remove inoculated set v = 0 and Pd = 0
N = 300 #TotalPopulationSize
alpha = 0.1 #InitialProportionExposed
v = 0.3 #InitialProportionInoculated #set alpha to 0 for no innocualted
C = 3 # ratio fake news spreads more than debunking
delta = 0.04 #RateOfInoculationSpreadDecay - rate at which innoculation effect reduced
E = 0.8 #EffectivenessInoculation - how effective the inoculation is compared to normal
attitude_multiplier = 5
proportion_inoculated = 0.80

for num_run in range(number_runs):


    random.shuffle(Attitude_Centers)
    flows = [["Sus_to_Ino",0],["Ino_to_Ino",0],["Sus_to_Deb",0],["Ino_to_Deb",0],["Ino_to_Spr",0],["Sus_to_Spr",0],["Deb_to_Sus",0],["Spr_to_Sus",0]]

    C1,C2 = 0,0

    if C >= 1:
        C1 = 1
        C2 = 1/C
    else:
        C1 = C
        C2 = 1

    print("C,C1,C2 ",C,C1,C2)

    compartment_history_proportion = [
        ((1-v)*N*(1-(alpha)))/N,
        proportion_inoculated*(v * N * (1 - ((alpha) * (1 - E))))/N,
        (alpha * (1 - v) * N * C1 + (alpha * v * N * (1 - E)))/N,
        (1-proportion_inoculated)*(v * N * (1 - ((alpha) * (1 - E))))/N]


    compartment_history = []
    for i in range(len(compartment_history_proportion)):
        compartment_history.append([int(compartment_history_proportion[i]*N)])

    temp = 0
    for item in compartment_history:
        temp += item[0]
    compartment_history[0][0] = + compartment_history[0][0] + N-temp

    S_Index = 0
    Z_Index = 1
    X_Index = 2
    R_Index = 3

    #list of compartment groups
    agent_list = [{},{},{},{}]

    #create the initial actors
    agent_num = 0
    def attitude_generator():
        return distributions.normal(-0.098027,0.49318)

    attitude_list = []
    for i in range(N):
        attitude_list.append(attitude_generator())

    if Use_Scale_Free_And_Centers:
        Node_List = []

        t1 = list(range(N))
        t2 = [None] * N
        t3 = [None] * N

        for i in range(len(t1)):
            Node_List.append([t1[i],t2[i],t3[i]]) #node id, node attitude, node role


        #can replace the following 2 lines with any networx undirected graph, no multigraph, no self loops
        main_network = networkx.Graph(networkx.scale_free_graph(N, alpha=0.1, beta=0.8, gamma=0.1, create_using=None, seed=None))
        main_network.remove_edges_from(networkx.selfloop_edges(main_network))

        Alternate_Attitude_Centers = []
        use_alternate = False

        edge_list = []
        for node in range(N):
            edge_list.append([len(main_network.edges(node)), node])
        edge_list.sort(reverse=True)
        #print(edge_list)
        print("average edges a node has: " + str ( sum(i[0] for i in edge_list) / N))
        Attitude_center_nodes = []
        list_of_bfs = []

        for i in range(len(Attitude_Centers)):
            attitude_list.pop()
            if use_alternate:
                Attitude_center_nodes.append(Alternate_Attitude_Centers[i])
            else:
                Attitude_center_nodes.append(edge_list[i][0])
            Node_List[Attitude_center_nodes[i]][1] = Attitude_Centers[i]
            bfs = list(networkx.bfs_successors(main_network, source=edge_list[i][0]))
            list_of_bfs.append([bfs[0][0], []])
            for j in range(len(bfs)):
                for k in range(len(bfs[j][1])):
                    list_of_bfs[i][1].append(bfs[j][1][k])

        breadth_first_index_list = [0] * len(list_of_bfs)

        def find_closest_index(array, number):
            array = np.asarray(array)
            idx = (np.abs(array - number)).argmin()
            return array[idx]

        while len(attitude_list) > 0:
            for i in range(len(Attitude_Centers)):
                for j in range(breadth_first_index_list[i],len(list_of_bfs[i][1])):
                    breadth_first_index_list[i] = j
                    if Node_List[list_of_bfs[i][1][j]][1] == None:
                        temp = find_closest_index(attitude_list,Attitude_Centers[i])
                        Node_List[list_of_bfs[i][1][j]][1] = temp
                        attitude_list.pop(attitude_list.index(temp))
                        break
                if len(attitude_list) == 0:
                    break

        for i in range(len(Node_List)):

            if Node_List[i][1] is None:
                print("isolate given")
                Node_List[i][1] = attitude_generator()
                print(Node_List[i][1])

        print("list of breadth first search: " + str(list_of_bfs))
        print("Attitude center nodes: " + str(Attitude_center_nodes))
        print("Attitude centers: " + str(Attitude_Centers))

        random.shuffle(Node_List)

        if inoculation_selection == 0:
            pass
        elif inoculation_selection == 1:
            Node_List.sort(key=lambda x: x[1], reverse=True)
        elif inoculation_selection == 2:
            Node_List.sort(key = lambda x: x[1])
            for i in range(int((N/2) - (compartment_history[Z_Index][0]/2))):
                Node_List.append(Node_List.pop(0))
        elif inoculation_selection == 3:
            Node_List.sort(key=lambda x: x[1])
        else:
            print("invalid inoculation selection")


        for i in range(compartment_history[Z_Index][0]):
            temp = Node_List.pop()
            agent_list[Z_Index][temp[0]] = agent("Z",temp[0],temp[1])
            agent_list[Z_Index][temp[0]].day_became_innoculated = 0
            agent_num += 1

        random.shuffle(attitude_list)

        for i in range(compartment_history[S_Index][0]):
            temp = Node_List.pop()
            agent_list[S_Index][temp[0]] = agent("S",temp[0],temp[1])
            agent_num += 1

        for i in range(compartment_history[X_Index][0]):
            temp = Node_List.pop()
            agent_list[X_Index][temp[0]] = agent("X",temp[0],temp[1])
            agent_num += 1

        for i in range(compartment_history[R_Index][0]):
            temp = Node_List.pop()
            agent_list[R_Index][temp[0]] = agent("R",temp[0],temp[1])
            agent_num += 1


    else:
        main_network = networkx.gnm_random_graph(N, N * (N - 1) / 2 * density)

        if inoculation_selection == 0:
            pass
        elif inoculation_selection == 1:
            attitude_list.sort(reverse=True)
        elif inoculation_selection == 2:
            attitude_list.sort()
            for i in range(int((N/2) - (compartment_history[Z_Index][0]/2))):
                attitude_list.append(attitude_list.pop(0))
        elif inoculation_selection == 3:
            attitude_list.sort()
        else:
            print("invalid inoculation selection")

        print(attitude_list)

        for i in range(compartment_history[Z_Index][0]):
            agent_list[Z_Index][agent_num] = agent("Z", agent_num, attitude_list.pop())
            agent_list[Z_Index][agent_num].day_became_innoculated = 0
            agent_num += 1

        random.shuffle(attitude_list)

        for i in range(compartment_history[S_Index][0]):
            agent_list[S_Index][agent_num] = agent("S",agent_num,attitude_list.pop())
            agent_num += 1

        for i in range(compartment_history[X_Index][0]):
            agent_list[X_Index][agent_num] = agent("X",agent_num,attitude_list.pop())
            agent_num += 1

        for i in range(compartment_history[R_Index][0]):
            agent_list[R_Index][agent_num] = agent("R",agent_num,attitude_list.pop())
            agent_num += 1


    #function to add a new iteration to the history
    def update_values(iteration,flows):

        for k in range(agent_num):
            agent_1_num = k
            agent_2_num_list = []
            agent_1 = 0
            agent_2 = []


            for l in range(len(main_network.edges(k))):
                agent_2_num_list.append(list(main_network.edges(k))[l][1])
                for group in range(len(agent_list)):
                    if agent_2_num_list[l] in agent_list[group]:
                        agent_2.append(group)
            for group in range(len(agent_list)):
                if agent_1_num in agent_list[group]:
                    agent_1 = group

            attitude = agent_list[agent_1][k].attitude

            attitude_become_spreader = 1
            attitude_become_debunker = 1

            if attitude >= 0:
                attitude_become_spreader = 1 + (attitude**2*attitude_multiplier)
                attitude_become_debunker = 1 - (attitude**2)
            elif attitude < 0:
                attitude_become_spreader = 1 - (attitude**2)
                attitude_become_debunker = 1 + (attitude**2*attitude_multiplier)

            #if attitude > 0:
            #    attitude_become_spreader = attitude_multiplier * attitude
            #    attitude_become_debunker = 1/attitude_multiplier * attitude
            #elif attitude < 0:
            #    attitude_become_spreader = 1/attitude_multiplier * -attitude
            #    attitude_become_debunker = attitude_multiplier * -attitude

            if agent_1 == 0:
                if len(agent_2) == 0:
                    #chance = Pd * (b) * attitude_become_debunker
                    chance = 0
                else:
                    #chance = Pd * (b + (C2 * agent_2.count(3) / len(agent_2))) * attitude_become_debunker
                    chance =  (C2 * agent_2.count(3) / len(agent_2)) * attitude_become_debunker
                if random.random() < chance:
                    if random.random() < proportion_inoculated + (agent_list[0][agent_1_num].attitude):
                        agent_list[1][agent_1_num] = agent_list[0][agent_1_num]
                        del(agent_list[0][agent_1_num])
                        agent_list[1][agent_1_num].status = "Z"
                        agent_list[1][agent_1_num].day_became_innoculated = iteration
                        agent_1 = 1
                        flows[0][1] += 1


                    else:
                        agent_list[3][agent_1_num] = agent_list[0][agent_1_num]
                        del(agent_list[0][agent_1_num])
                        agent_list[3][agent_1_num].status = "R"
                        agent_1 = 3
                        flows[2][1] += 1

            if agent_1 == 1:
                if len(agent_2) == 0:
                    #chance = Pd * (b) * attitude_become_debunker
                    chance = 0
                else:
                    #chance = Pd * (b + (C2 * agent_2.count(3) / len(agent_2))) * attitude_become_debunker
                    chance =  (C2 * agent_2.count(3) / len(agent_2)) * attitude_become_debunker
                if random.random() < chance:
                    if random.random() < proportion_inoculated + (agent_list[1][agent_1_num].attitude):
                        agent_list[1][agent_1_num].day_became_innoculated = iteration
                        flows[1][1] += 1
                    else:
                        agent_list[3][agent_1_num] = agent_list[1][agent_1_num]
                        del (agent_list[1][agent_1_num])
                        agent_list[3][agent_1_num].status = "R"
                        agent_1 = 3
                        flows[3][1] += 1

         #   elif agent_1 == 2:
         #       if len(agent_2) == 0:
         #           chance = Pd * (b) * attitude_become_debunker
         #       else:
         #           chance = Pd * (b + (C2 * agent_2.count(3) / len(agent_2))) * attitude_become_debunker
         #       if random.random() < chance:
         #           agent_list[3][agent_1_num] = agent_list[2][agent_1_num]
         #           del (agent_list[2][agent_1_num])
         #           agent_list[3][agent_1_num].status = "R"
         #           agent_1 = 3

            if agent_1 == 0:
                if len(agent_2) == 0:
                    chance = 0
                else:
                    chance =  C1 * agent_2.count(2) / len(agent_2) * attitude_become_spreader
                if random.random() < chance:
                    agent_list[2][agent_1_num] = agent_list[0][agent_1_num]
                    del (agent_list[0][agent_1_num])
                    agent_list[2][agent_1_num].status = "X"
                    agent_1 = 2
                    flows[5][1] += 1

            if agent_1 == 1:
                if len(agent_2) == 0:
                    chance = 0
                else:
                    chance = (C1*agent_2.count(2) / len(agent_2))*(1-(E*math.exp(-delta*(iteration-agent_list[1][agent_1_num].day_became_innoculated))))* attitude_become_spreader
                if random.random() < chance:
                    agent_list[2][agent_1_num] = agent_list[1][agent_1_num]
                    del (agent_list[1][agent_1_num])
                    agent_list[2][agent_1_num].status = "X"
                    agent_1 = 2
                    flows[4][1] += 1

            if agent_1 == 2:
                if len(agent_2) == 0:
                    chance = 0
                else:
                    chance = (agent_2.count(3) / len(agent_2)) * attitude_become_debunker
                if random.random() < chance:
                    agent_list[0][agent_1_num] = agent_list[2][agent_1_num]
                    del (agent_list[2][agent_1_num])
                    agent_list[0][agent_1_num].status = "S"
                    agent_1 = 0
                    flows[6][1] += 1

            if agent_1 == 3:
                if len(agent_2) == 0:
                    chance = 0
                else:
                    chance = (agent_2.count(2) / len(agent_2)) * attitude_become_spreader
                if random.random() < chance:
                    agent_list[0][agent_1_num] = agent_list[3][agent_1_num]
                    del (agent_list[3][agent_1_num])
                    agent_list[0][agent_1_num].status = "S"
                    agent_1 = 0
                    flows[7][1] += 1


            else:
                pass

        compartment_history[S_Index].append(len(agent_list[S_Index]))
        compartment_history[Z_Index].append(len(agent_list[Z_Index]))
        compartment_history[X_Index].append(len(agent_list[X_Index]))
        compartment_history[R_Index].append(len(agent_list[R_Index]))

    for i in range(iterations):

        if i % 20 == 0 and make_network_image:
            Node_Colour = ['yellow'] * agent_num

            for group in range(len(agent_list)):
                for node in agent_list[group]:

                    if group == R_Index:
                        Node_Colour[agent_list[R_Index][node].number] = 'blue'
                    if group == Z_Index:
                        Node_Colour[agent_list[Z_Index][node].number] = 'orange'
                    if group == S_Index:
                        Node_Colour[agent_list[S_Index][node].number] = 'green'
                    if group == X_Index:
                        Node_Colour[agent_list[X_Index][node].number] = 'red'



            networkx.draw(main_network, node_size=100, node_color = Node_Colour)#, with_labels=True)
            plt.savefig(str(i) + '_network.png')
            #plt.show()
            plt.close()

        if i % 20 == 0 and make_attitude_spread:

            print("Iteration: " + str(i))
            for j in range(len(flows)):
                print(str(flows[j][0])+": " + str(flows[j][1]))


            bins_R,bins_X,bins_S,bins_Z = [],[],[],[]
            figure,axis = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

            for a in range(2):
                for b in range(2):
                    axis[a,b].set_xlim(-1, 1)
                    axis[a, b].set_ylim(0, N/5)

            average_att = 0
            for group in range(len(agent_list)):
                for node in agent_list[group]:

                    if group == R_Index:
                        bins_R.append(agent_list[R_Index][node].attitude)
                        average_att += agent_list[R_Index][node].attitude
                    if group == X_Index:
                        bins_X.append(agent_list[X_Index][node].attitude)
                        average_att += agent_list[X_Index][node].attitude
                    if group == S_Index:
                        bins_S.append(agent_list[S_Index][node].attitude)
                        average_att += agent_list[S_Index][node].attitude
                    if group == Z_Index:
                        bins_Z.append(agent_list[Z_Index][node].attitude)
                        average_att += agent_list[Z_Index][node].attitude

            print("Average attitude: " + str(int(average_att/N*1000)/1000))
            print()
            axis[0,0].hist(bins_S, bins =10)
            axis[0, 1].hist(bins_X, bins=10)
            axis[1, 0].hist(bins_Z, bins=10)
            axis[1, 1].hist(bins_R, bins=10)

            axis[0, 0].set(title='Susceptible', ylabel='Frequency')
            axis[0, 1].set(title='Misinformation Spreaders', ylabel='Frequency')
            axis[1, 0].set(title='Inoculated', ylabel='Frequency')
            axis[1,1].set(title='Debunkers', ylabel='Frequency')




            plt.savefig(str(i) + '_attitude.png')
            #plt.show()
            plt.close()

        update_values(i,flows)

    #draw lines on chart and display
    print(compartment_history[X_Index][-1],compartment_history[R_Index][-1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(compartment_history[R_Index],label='Debunker')
    plt.plot(compartment_history[Z_Index],label='Inoculated')
    plt.plot(compartment_history[S_Index],label='Susceptible')
    plt.plot(compartment_history[X_Index],label='Misinformation Spreader')
    plt.legend()
    ax.set_ylabel('Agents')
    ax.set_xlabel('Days')
    ax.set_title('SZXR compartment model with agents')
    plt.savefig(str(iterations) + '_graph.png')
    plt.show()

    def test_distribution():
        temp = []
        for i in range(10000):
            temp.append(attitude_generator())

        plt.hist(temp, bins = 100)
        plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
        plt.show()

    #test_distribution()


