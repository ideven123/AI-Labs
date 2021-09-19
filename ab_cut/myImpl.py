import util

"""
Data sturctures we will use are stack, queue and priority queue.
Stack: first in last out
Queue: first in first out
    collection.push(element): insert element
    element = collection.pop() get and remove element from collection

Priority queue:
    pq.update('eat', 2)
    pq.update('study', 1)
    pq.update('sleep', 3)
pq.pop() will return 'study' because it has highest priority 1.

"""

"""
problem is a object has 3 methods related to search state:

problem.getStartState()
Returns the start state for the search problem.

problem.isGoalState(state)
Returns True if and only if the state is a valid goal state.

problem.getChildren(state)
For a given state, this should return a list of tuples, (next_state,
step_cost), where 'next_state' is a child to the current state, 
and 'step_cost' is the incremental cost of expanding to that child.

"""
def myDepthFirstSearch(problem):
    visited = {}
    frontier = util.Stack()
    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]                
        
        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []

def myBreadthFirstSearch(problem):
    visited = {}
    frontier = util.Queue()
    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state
            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))
    return []

def myAStarSearch(problem, heuristic):
    visited = {}
    close_list = []
    state = problem.getStartState()
    open_list = [state]
    if  not len(open_list):
        return []
    h = {}
    g = {}
    f = {}
    h[state] = heuristic(state)
    g[state] = 0
    f[state] = h[state] + g[state]
    visited[state] = None
    while len(open_list):
        #在open_list 里找最好的估计 ， 记为 best_node
        min_state  = open_list[0]
        for state in open_list:
            if(f[state] < f[min_state]):
                min_state = state
        #如果结束.则输出        
        if(problem.isGoalState(min_state)):
            solution = [min_state]
            prev_state = visited[min_state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]
        #便利best_node 的子节点
        for child_state, step_cost in problem.getChildren(min_state):
            child_state_g = step_cost + g[min_state]
            if(child_state in close_list):
                continue
            elif(child_state in open_list):
                if(child_state_g < g[child_state]):
                    g[child_state] = child_state_g
                    f[child_state] = g[child_state] + h[child_state]
                    visited[child_state]  = min_state
            else:
                open_list.append(child_state)
                g[child_state] = child_state_g
                h[child_state] = heuristic(child_state)
                f[child_state] = g[child_state] + h[child_state]
                visited[child_state] = min_state
        #最小的加到close
        close_list.append(min_state)
        open_list.remove(min_state)
        del f[min_state]
        del g[min_state]
        del h[min_state]
        # 检测best_node 是否最终状态
    return []

"""
Game state has 4 methods we can use.

state.isTerminated()
Return True if the state is terminated. We should not continue to search if the state is terminated.

state.isMe()
Return True if it's time for the desired agent to take action. We should check this function to determine whether an agent should maximum or minimum the score.

state.getChildren()
Returns a list of legal state after an agent takes an action.

state.evaluateScore()
Return the score of the state. We should maximum the score for the desired agent.

"""
class MyMinimaxAgent():

    def __init__(self, depth):
        self.depth = depth

    def minimax(self, state, depth):
        if state.isTerminated():
            return None, state.evaluateScore()    

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')
        for child in state.getChildren():
            if child.isMe():
                if depth==1 :
                    child_score = child.evaluateScore()        
                else:
                     _,child_score = self.minimax(child , depth-1)
            else:
                 _,child_score = self.minimax(child , depth)

            if state.isMe(): 
                    #找最大的
                if child_score > best_score:
                    best_state = child
                    best_score = child_score
            else:
                    #找最小的
                if child_score < best_score:
                    best_state = child
                    best_score = child_score
        return best_state, best_score

    def getNextState(self, state):
        best_state, _ = self.minimax(state, self.depth)
        return best_state

class MyAlphaBetaAgent():

    def __init__(self, depth):
        self.depth = depth

    def AlphaBeta(self,state,depth,a,b):
        if( state.isTerminated()):
            return None, state.evaluateScore()
        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')
        for child in state.getChildren():
            # state 极大值点 。 找最大的子节点 ， 是作为 a 的最小值
            if child.isMe():
                if depth==1 :
                    child_score = child.evaluateScore()        
                else:
                     _,child_score = self.AlphaBeta(child , depth-1,a,b)
            else:
                 _,child_score = self.AlphaBeta(child , depth,a,b)
            if state.isMe():
                if( child_score > best_score ):
                    best_state = child
                    best_score = child_score
                if best_score > b :    
                    break       #剪枝
                a = max(a,best_score)
            else: 
                # state 极小值点 。 找最小的子节点 ， 是作为 b 的最大值
                if( child_score < best_score ):
                    best_state = child
                    best_score = child_score
                if best_score  < a:    
                    break       #剪枝
                b = min(b,best_score)                
        return best_state,best_score       
    def getNextState(self, state):
        a = -float('inf')
        b = float('inf')
        best_state, _ = self.AlphaBeta(state, self.depth,a,b)
        return best_state