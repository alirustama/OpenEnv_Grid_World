"""Unit tests for OpenEnv environment"""

import pytest
from env.environment import MyOpenEnvEnvironment
from env.graders import TaskGrader, MetricsTracker
from env.tasks import EasyNavigation, MediumNavigation, HardNavigation


class TestEnvironment:
    """Test environment functionality"""
    
    @pytest.fixture
    def env(self):
        """Create environment instance"""
        return MyOpenEnvEnvironment()
    
    def test_reset(self, env):
        """Test environment reset"""
        obs = env.reset()
        
        assert "agent_position" in obs
        assert "target_position" in obs
        assert "distance_to_target" in obs
        assert obs["step_count"] == 0
        assert obs["episode_reward"] == 0.0
    
    def test_step(self, env):
        """Test step execution"""
        env.reset()
        
        action = {"direction": "up", "magnitude": 0.5}
        obs, reward, done = env.step(action)
        
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert obs["step_count"] == 1
    
    def test_agent_movement(self, env):
        """Test agent moves in correct direction"""
        env.reset()
        initial_pos = env.agent_pos.copy()
        
        # Move up
        action = {"direction": "up", "magnitude": 0.5}
        env.step(action)
        assert env.agent_pos[1] > initial_pos[1]
    
    def test_step_limit(self, env):
        """Test episode ends at max steps"""
        env.reset()
        env.max_steps = 10
        
        for _ in range(12):
            _, _, done = env.step({"direction": "up", "magnitude": 0.5})
            if env.step_count >= env.max_steps:
                assert done
                break
    
    def test_target_collision(self, env):
        """Test detection of target reach"""
        env.reset()
        env.target_pos = env.agent_pos.copy()
        
        _, reward, _ = env.step({"direction": "up", "magnitude": 0.5})
        # When target is reached, should get positive reward
        assert reward > 0


class TestTasks:
    """Test task definitions"""
    
    def test_easy_task(self):
        """Test easy navigation task"""
        task = EasyNavigation()
        
        assert task.name == "easy_navigation"
        assert task.difficulty == 1
        assert task.max_steps == 150  # Updated for stochastic config
    
    def test_medium_task(self):
        """Test medium navigation task"""
        task = MediumNavigation()
        
        assert task.name == "medium_navigation"
        assert task.difficulty == 2
        assert task.max_steps == 200  # Updated for stochastic config
    
    def test_hard_task(self):
        """Test hard navigation task"""
        task = HardNavigation()
        
        assert task.name == "hard_navigation"
        assert task.difficulty == 3
        assert task.max_steps == 150  # Updated for stochastic config
        
        assert task.name == "hard_navigation"
        assert task.difficulty == 3
    
    def test_task_scoring(self):
        """Test task scoring logic"""
        task = EasyNavigation()
        
        # High reward = high score
        score_high = task.calculate_score(10.0, 50)
        score_low = task.calculate_score(1.0, 50)
        
        assert score_high > score_low


class TestGrader:
    """Test grading system"""
    
    @pytest.fixture
    def grader(self):
        """Create grader instance"""
        return TaskGrader()
    
    def test_grade_trajectory(self, grader):
        """Test trajectory grading"""
        trajectory = {
            "episode_reward": 10.0,
            "steps_taken": 50,
            "completed": True
        }
        
        score = grader.grade_trajectory("easy_navigation", trajectory)
        
        assert 0.0 <= score <= 1.0
    
    def test_grade_incomplete(self, grader):
        """Test grading incomplete trajectory"""
        trajectory = {
            "episode_reward": 2.0,
            "steps_taken": 100,
            "completed": False,
            "partial_reward": 2.0
        }
        
        score = grader.grade_trajectory("easy_navigation", trajectory)
        assert score <= 0.5


class TestMetricsTracker:
    """Test metrics tracking"""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker instance"""
        return MetricsTracker()
    
    def test_record_step(self, tracker):
        """Test recording steps"""
        tracker.record_step(distance=10, reward=0.5, collision=False)
        
        assert tracker.steps == 1
        assert tracker.total_reward == 0.5
        assert tracker.collisions == 0
    
    def test_collision_tracking(self, tracker):
        """Test collision recording"""
        tracker.record_step(distance=5, reward=-1.0, collision=True)
        
        assert tracker.collisions == 1
    
    def test_summary(self, tracker):
        """Test summary generation"""
        tracker.record_step(distance=10, reward=0.5)
        tracker.record_step(distance=5, reward=0.8)
        tracker.set_target_reached()
        
        summary = tracker.get_summary()
        
        assert summary["steps"] == 2
        assert summary["total_reward"] == 1.3
        assert summary["target_reached"] == True


@pytest.mark.integration
class TestIntegration:
    """Integration tests"""
    
    def test_full_episode(self):
        """Test complete episode execution"""
        env = MyOpenEnvEnvironment()
        env.reset()
        
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = {"direction": "up", "magnitude": 0.8}
            obs, reward, done = env.step(action)
            steps += 1
        
        assert steps > 0
        assert obs["step_count"] == steps
    
    def test_all_tasks_scoreable(self):
        """Test all tasks can be graded"""
        grader = TaskGrader()
        
        for task_name in ["easy_navigation", "medium_navigation", "hard_navigation"]:
            trajectory = {
                "episode_reward": 5.0,
                "steps_taken": 50,
                "completed": True
            }
            
            score = grader.grade_trajectory(task_name, trajectory)
            assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
